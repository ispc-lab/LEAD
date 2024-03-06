import os
import sys 
import logging 

import torch 
import faiss 
import random 

import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm 

def set_random_seed(seed=0):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def log_args(args):
    s = "\n==========================================\n"
    
    s += ("python" + " ".join(sys.argv) + "\n")
    
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    
    s += "==========================================\n"
    
    return s

def set_logger(args, log_name="train_log.txt"):
    
    log_format = "%(asctime)s [%(levelname)s] - %(message)s"
    # creating logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # file logger handler
    if args.test:
        # Append the test results on existing logging file.
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="a")
        file_format = logging.Formatter("%(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
    else:
        # Init the logging file.
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="w")
        
        file_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
    
    # terminal logger handler
    terminal_handler = logging.StreamHandler()
    terminal_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(terminal_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)
    if not args.test:
        logger.debug(log_args(args))
    
    return logger

def compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flag=True, open_thresh=0.5, pred_unc_all=None):
    
    # class_list:
    #   :source [0, 1, ..., N_share - 1, ...,           N_share + N_src_private - 1]
    #   :target [0, 1, ..., N_share - 1, N_share + N_src_private + N_tar_private -1]
    # gt_label_all [N]
    # pred_cls_all [N, C]
    # open_flag    True/False
    # pred_unc_all [N], if exists. [0~1.0]
    
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = torch.max(pred_cls_all, dim=1)[1] #[N]
    
    if open_flag:
        cls_num = pred_cls_all.shape[1]
        
        if pred_unc_all is None:
            # If there is not pred_unc_all tensor,
            # We normalize the Shannon entropy to [0, 1] to denote the uncertainty.
            pred_unc_all = Entropy(pred_cls_all)/np.log(cls_num)# [N]

        unc_idx = torch.where(pred_unc_all > open_thresh)[0]
        pred_label_all[unc_idx] = cls_num # set these pred results to unknown

    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0

    return h_score, known_acc, unknown_acc, per_class_acc


class CrossEntropyLabelSmooth(nn.Module):
    
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """      

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets, applied_softmax=True):
        """
        Args:
            inputs: prediction matrix (after softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes).
        """
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)
        
        if inputs.shape != targets.shape:
            # this means that the target data shape is (B,)
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
         
        if self.reduction:
            return loss.mean()
        else:
            return loss