import torch
from torch.utils.data import DataLoader
from utils.process_utils import get_process_pipeline,get_act
from utils.wsi_utils import WSI_Dataset, WSI_Dataset_modified
from utils.general_utils import set_global_seed,init_epoch_info_log,add_epoch_info_log,early_stop
from utils.model_utils import get_optimizer,get_scheduler,get_criterion,save_last_model,save_log,model_select
from utils.loop_utils import clam_train_loop,clam_val_loop, clam_train_loop_modified, clam_val_loop_modified
from tqdm import tqdm
from modules.CLAM_MB_MIL.clam_mb_mil_modified import CLAM_MB_MIL_modified
from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL

    
def process_CLAM_MB_MIL(args):
    legacy_modified = args.General.modified
    if legacy_modified:
        train_dataset = WSI_Dataset_modified(args.Dataset.dataset_csv_path,'train')
        train_dataset.clustering_pseudo_labeling(**dict(args.Dataset.pseudo_labeling))
        val_dataset = WSI_Dataset_modified(args.Dataset.dataset_csv_path,'val')
        test_dataset = WSI_Dataset_modified(args.Dataset.dataset_csv_path,'test')
        process_pipeline = 'Train_Val_Test'
        args.General.process_pipeline = process_pipeline
    else:
        train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'train')
        val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'val')
        test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'test')
        process_pipeline = get_process_pipeline(val_dataset,test_dataset) 
        args.General.process_pipeline = process_pipeline
    '''
    generator settings
    '''
    
    generator = torch.Generator()
    generator.manual_seed(args.General.seed) 
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    use_balanced_sampler = args.Dataset.balanced_sampler.use
    if use_balanced_sampler:
        sampler = train_dataset.get_balanced_sampler(replacement = args.Dataset.balanced_sampler.replacement)
        if legacy_modified:
            train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers = num_workers,generator=generator,sampler=sampler, collate_fn=lambda batch: batch[0])
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers = num_workers,generator=generator,sampler=sampler)
    else:
        if legacy_modified:
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = num_workers,generator=generator, collate_fn=lambda batch: batch[0])
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = num_workers,generator=generator)
    
    if legacy_modified:
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: batch[0])
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=lambda batch: batch[0])
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    print('DataLoader Ready!')
    in_dim = args.Model.in_dim
    subtyping = args.Model.subtyping
    k_sample = args.Model.k_sample
    size_arg = args.Model.size_arg
    dropout = args.Model.dropout
    num_classes = args.General.num_classes
    gate = args.Model.gate
    act = get_act(args.Model.act)
    instance_eval = args.Model.instance_eval
    device = torch.device(f'cuda:{args.General.device}')
    instance_loss_fn = args.Model.instance_loss_fn
    instance_loss_fn = get_criterion(instance_loss_fn)
    bag_weight = args.Model.bag_weight

    if legacy_modified:
        pseudo_loss_weight = args.Model.pseudo_loss_weight
        pseudo_criterion = train_dataset.criterion
    if legacy_modified:
        mil_model = CLAM_MB_MIL_modified(gate, size_arg, dropout, k_sample, num_classes, instance_loss_fn, subtyping, in_dim,act,instance_eval)
    else:
        mil_model = CLAM_MB_MIL(gate, size_arg, dropout, k_sample, num_classes, instance_loss_fn, subtyping, in_dim,act,instance_eval)
    mil_model.to(device)
    
    print('Model Ready!')
    
    optimizer,base_lr = get_optimizer(args,mil_model)
    if args.Model.scheduler.use:
        scheduler,warmup_scheduler = get_scheduler(args,optimizer,base_lr)
    else:
        scheduler,warmup_scheduler = None,None
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    '''
    begin training
    '''
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = float('-inf')
    if best_model_metric == 'val_loss':
        REVERSE = True
        best_val_metric = 9999
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:',process_pipeline)
    for epoch in tqdm(range(args.General.num_epochs),colour='GREEN'):
        if epoch+1 <= warmup_epoch:
            now_scheduler = warmup_scheduler
        else:
            now_scheduler = scheduler
        if legacy_modified:
            train_loss,cost_time = clam_train_loop_modified(device,mil_model,train_dataloader,criterion,pseudo_criterion,optimizer,now_scheduler,bag_weight,pseudo_loss_weight)
        else:
            train_loss,cost_time = clam_train_loop(device,mil_model,train_dataloader,criterion,optimizer,now_scheduler,bag_weight)
        if process_pipeline == 'Train_Val_Test':
            
            if legacy_modified:
                val_loss,val_metrics = clam_val_loop_modified(device,num_classes,mil_model,val_dataloader,criterion,bag_weight)
                test_loss,test_metrics = clam_val_loop_modified(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
            else:
                val_loss,val_metrics = clam_val_loop(device,num_classes,mil_model,val_dataloader,criterion,bag_weight)
                test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
        elif process_pipeline == 'Train_Val':
            if legacy_modified:
                val_loss,val_metrics = clam_val_loop_modified(device,num_classes,mil_model,val_dataloader,criterion,bag_weight)
            else:
                val_loss,val_metrics = clam_val_loop(device,num_classes,mil_model,val_dataloader,criterion,bag_weight)

            test_loss,test_metrics = None,None
        elif process_pipeline == 'Train_Test':
            val_loss,val_metrics,test_loss,test_metrics = None,None,None,None
            if epoch+1 == args.General.num_epochs:
                if legacy_modified:
                    test_loss,test_metrics = clam_val_loop_modified(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
                else:
                    test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)

        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics)
        
        # model selection, it only works when process_pipeline is 'Train_Val_Test' or 'Train_Val'
        best_val_metric,best_epoch = model_select(REVERSE,args,mil_model.state_dict(),val_metrics,best_model_metric,best_val_metric,epoch,best_epoch)

        '''
        early stop
        '''
        if early_stop(args,epoch_info_log,process_pipeline,epoch,mil_model.state_dict(),best_epoch):
            break

        if epoch+1 == args.General.num_epochs:
            save_last_model(args,mil_model.state_dict(),epoch+1)
            save_log(args,epoch_info_log,best_epoch,process_pipeline)
