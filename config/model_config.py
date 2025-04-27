import os 
import argparse

def build_args():
    
    parser = argparse.ArgumentParser("This script is used to achieve Source-free Universal Domain Adaptation based on our LEAD framework")
    
    parser.add_argument("--dataset", type=str, default="Office")
    parser.add_argument("--backbone_arch", type=str, default="resnet50", help="resnet50, resnet101, vgg19")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=1)

    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--epochs", default=50, type=int)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--lam_psd", default=0.3, type=float)
    parser.add_argument("--w_0", default=0.55, type=float)

    parser.add_argument("--source_train_type", default="smooth", type=str, help="vanilla, smooth")
    parser.add_argument("--target_label_type", default="OPDA", type=str)
    parser.add_argument("--target_private_class_num", default=None, type=int)
    parser.add_argument("--note", default=None)
    
    args = parser.parse_args()
        
    '''
    assume classes across domains are the same.
    [0 1 ............................................................................ N - 1]
    |---- common classes --||---- source private classes --||---- target private classes --|

    |-------------------------------------------------|
    |                DATASET PARTITION                |
    |-------------------------------------------------|
    |DATASET    |  class split(com/sou_pri/tar_pri)   |
    |-------------------------------------------------|
    |DATASET    |    PDA    |    OSDA    | OPDA/UniDA |
    |-------------------------------------------------|
    |Office-31  |  10/21/0  |  10/0/11   |  10/10/11  |
    |-------------------------------------------------|
    |OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
    |-------------------------------------------------|
    |VisDA-C    |   6/6/0   |   6/0/6    |   6/3/3    |
    |-------------------------------------------------|  
    |DomainNet  |           |            | 150/50/145 |
    |-------------------------------------------------|
    '''
    
    if args.dataset == "Office":
        domain_list = ['Amazon', 'Dslr', 'Webcam']
        args.source_data_dir = os.path.join("./data/Office", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("./data/Office", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("./data/Office", item) for item in args.target_domain_list]
         
        args.shared_class_num = 10
        
        if args.target_label_type == "PDA":
            args.source_private_class_num = 21
            args.target_private_class_num = 0
        
        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 11
            
        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 10
            if args.target_private_class_num is None:
                args.target_private_class_num = 11
        
        else:
            raise NotImplementedError("Unknown target label type specified")
 
    elif args.dataset == "OfficeHome":
        domain_list = ['Art', 'Clipart', 'Product', 'Realworld']
        args.source_data_dir = os.path.join("./data/OfficeHome", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("./data/OfficeHome", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(4) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("./data/OfficeHome", item) for item in args.target_domain_list]
        
        if args.target_label_type == "PDA":
            args.shared_class_num = 25
            args.source_private_class_num = 40
            args.target_private_class_num = 0
            
        elif args.target_label_type == "OSDA":
            args.shared_class_num = 25
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 40
        
        elif args.target_label_type == "OPDA":
            args.shared_class_num = 10
            args.source_private_class_num = 5
            if args.target_private_class_num is None:
                args.target_private_class_num = 50
        else:
            raise NotImplementedError("Unknown target label type specified")

    elif args.dataset == "VisDA":
        args.source_data_dir = "./data/VisDA/train/"
        args.target_data_dir = "./data/VisDA/validation/"
        args.target_domain_list = ["validataion"]
        args.target_domain_dir_list = [args.target_data_dir]
        
        args.shared_class_num = 6
        if args.target_label_type == "PDA":
            args.source_private_class_num = 6
            args.target_private_class_num = 0
        
        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            args.target_private_class_num = 6
        
        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 3
            args.target_private_class_num = 3
            
        else:
            raise NotImplementedError("Unknown target label type specified", args.target_label_type)
    
    elif args.dataset == "DomainNet":
        domain_list = ["Painting", "Real", "Sketch"]
        args.source_data_dir = os.path.join("./data/DomainNet", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("./data/DomainNet", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("./data/DomainNet", item) for item in args.target_domain_list]

        args.embed_feat_dim = 1024
        args.shared_class_num = 150
        if args.target_label_type == "OPDA":
            args.source_private_class_num = 50
            args.target_private_class_num = 145
        else:
            raise NotImplementedError("Unknown target label type specified")
        
    args.source_class_num = args.shared_class_num + args.source_private_class_num
    args.target_class_num = args.shared_class_num + args.target_private_class_num
    args.class_num = args.source_class_num
    
    args.source_class_list = [i for i in range(args.source_class_num)]
    args.target_class_list = [i for i in range(args.shared_class_num)]
    if args.target_private_class_num > 0:
        args.target_class_list.append(args.source_class_num)

    return args
