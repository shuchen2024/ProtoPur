import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB_MIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    @staticmethod
    def create_targets(length, target, device):
        return torch.full((length, ), target, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    # def inst_eval(self, A, h, classifier): 
    #     device=h.device
    #     if len(A.shape) == 1:
    #         A = A.view(1, -1)
    #     top_p_ids = torch.topk(A, self.k_sample)[1][-1]
    #     top_p = torch.index_select(h, dim=0, index=top_p_ids)
    #     top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
    #     top_n = torch.index_select(h, dim=0, index=top_n_ids)
    #     p_targets = self.create_positive_targets(self.k_sample, device)
    #     n_targets = self.create_negative_targets(self.k_sample, device)

    #     all_targets = torch.cat([p_targets, n_targets], dim=0)
    #     all_instances = torch.cat([top_p, top_n], dim=0)
    #     logits = classifier(all_instances)
    #     all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
    #     instance_loss = self.instance_loss_fn(logits, all_targets)
    #     return instance_loss, all_preds, all_targets
    
    def inst_eval(self, A, h, target, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_targets(self.k_sample, target, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        classifier.requires_grad_(False)
        logits = classifier(all_instances)
        classifier.requires_grad_(True)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def inst_eval_pseudo(self, A, h, target, classifier, pseudo_labels): 
        device=h.device
        # One ranking keeps positive and negative selections disjoint, including ties.
        ranked_ids = torch.argsort(A.reshape(-1), descending=True, stable=True)
        n_positive = min(self.k_sample, len(ranked_ids))
        n_negative = min(self.k_sample, len(ranked_ids) - n_positive)
        top_p_ids = ranked_ids[:n_positive]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_p_pseudo_labels = torch.index_select(pseudo_labels, dim=0, index=top_p_ids)
        top_n_ids = ranked_ids[len(ranked_ids) - n_negative:]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        top_n_pseudo_labels = torch.index_select(pseudo_labels, dim=0, index=top_n_ids)
        p_targets = self.create_targets(n_positive, target, device)
        n_targets = self.create_negative_targets(n_negative, device)
        p_mask = (top_p_pseudo_labels == target) | (top_p_pseudo_labels == -1)
        n_mask = (top_n_pseudo_labels == 0) | (top_n_pseudo_labels == -1)

        all_targets = torch.cat([p_targets[p_mask], n_targets[n_mask]], dim=0)
        all_instances = torch.cat([top_p[p_mask], top_n[n_mask]], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        if all_targets.numel() == 0:
            instance_loss = logits.sum() * 0.0
        else:
            instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        classifier.requires_grad_(False)
        logits = classifier(top_p)
        classifier.requires_grad_(True)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets
    
    def inst_eval_out_pseudo(self, A, h, classifier, pseudo_labels):
        device=h.device
        top_p_ids = torch.argsort(A.reshape(-1), descending=True, stable=True)[:self.k_sample]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_p_pseudo_labels = torch.index_select(pseudo_labels, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(len(top_p_ids), device)
        # Unknown patches can be positives in the true-class branch of this shared head.
        p_mask = top_p_pseudo_labels == 0
        p_targets = p_targets[p_mask]
        logits = classifier(top_p[p_mask])
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        if p_targets.numel() == 0:
            instance_loss = logits.sum() * 0.0
        else:
            instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB_MIL_modified(CLAM_SB_MIL):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, num_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024,act=nn.ReLU(),instance_eval=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), act, nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = num_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = num_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(num_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        # instance_classifiers = [nn.Linear(size[1], 2) for i in range(num_classes)]
        # self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_classifiers = nn.Linear(size[1], num_classes+1)
        # self.instance_classifiers = nn.Linear(size[1], num_classes)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.num_classes = num_classes
        self.subtyping = subtyping
        self.act = act
        self.instance_eval = instance_eval

    def forward(self, h, pseudo_labels=None, pseudo_criterion=None, label=None, instance_eval=False, return_features=False, attention_only=False,return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        instance_eval = self.instance_eval
        if h.ndim == 3:
            h = h.squeeze(0)
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        forward_return['instance_loss'] = 0.0
        pseudo_loss = 0.0
        if pseudo_labels is not None:
            pseudo_labels = pseudo_labels.reshape(-1).long().to(h.device)
        if pseudo_labels is not None and pseudo_criterion is not None:
            pseudo_logits = self.instance_classifiers(h)
            if (pseudo_labels >= 0).any():
                pseudo_loss = pseudo_criterion(pseudo_logits, pseudo_labels)
            else:
                pseudo_loss = pseudo_logits.sum() * 0.0
        if instance_eval and label is not None and pseudo_labels is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze() #binarize label
            # for i in range(len(self.instance_classifiers)):
            for i in range(len(inst_labels)):
                inst_label = inst_labels[i].item()
                # classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval_pseudo(A[i], h, i+1, self.instance_classifiers, pseudo_labels)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out_pseudo(A[i], h, self.instance_classifiers, pseudo_labels)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                # total_inst_loss /= len(self.instance_classifiers)
                total_inst_loss /= self.num_classes
        
        M = torch.mm(A, h) 

        logits = torch.empty(1, self.num_classes).float().to(M.device)
        for c in range(self.num_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval and label is not None and pseudo_labels is not None:
            forward_return['instance_loss'] = total_inst_loss
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_raw[Y_hat]
        if return_WSI_feature:
            forward_return['WSI_feature'] = M[Y_hat].squeeze(0)
        forward_return['logits'] = logits
        forward_return['pseudo_loss'] = pseudo_loss
        # print('pseudo-labels_loss: ', forward_return['pseudo_loss'])
        return forward_return
