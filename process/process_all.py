from utils.general_utils import save_dataset_csv, print_args
from utils.yaml_utils import save_yaml


def process(args, yaml_path, options):
    save_dataset_csv(args)
    save_yaml(args, yaml_path, options)
    print_args(args)
    if args.General.MODEL_NAME == 'CLAM_MB_MIL':
        from .CLAM_MB_MIL.process_clam_mb_mil import process_CLAM_MB_MIL
        process_CLAM_MB_MIL(args)
    elif args.General.MODEL_NAME == 'DS_MIL':
        from .DS_MIL.process_ds_mil import process_DS_MIL
        process_DS_MIL(args)
    else:
        raise ValueError('This release supports CLAM_MB_MIL and DS_MIL only')
