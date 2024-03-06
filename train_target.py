import os
import faiss
import torch 
import shutil 
import numpy as np

from tqdm import tqdm 
from model.SFUniDA import SFUniDA 
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader

from config.model_config import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


best_score = 0.0
best_coeff = 1.0
@torch.no_grad()
def obtain_LEAD_pseudo_labels(args, model, dataloader, epoch_idx=0.0):
    
    model.eval()
    pred_cls_bank = []
    gt_label_bank = []
    embed_feat_bank = []
    
    class_list = args.target_class_list
    
    args.logger.info("Generating offline feat_decomposation based pseudo labels...")
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        embed_feat, pred_cls = model(imgs_test, apply_softmax=True)
        pred_cls_bank.append(pred_cls)
        embed_feat_bank.append(embed_feat)
        gt_label_bank.append(imgs_label.cuda())
    
    pred_cls_bank = torch.cat(pred_cls_bank, dim=0) #[N, C]
    gt_label_bank = torch.cat(gt_label_bank, dim=0) #[N]
    embed_feat_bank = torch.cat(embed_feat_bank, dim=0) #[N, D]
    embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)

    global best_score
    global best_coeff
    
    # Performing C_t estimation
    if epoch_idx % 10 == 0:
        args.logger.info("Performing C_t estimation...")
        coeff_list = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        embed_feat_bank_cpu = embed_feat_bank.cpu().numpy()
        
        if args.dataset == "VisDA":
            np.random.seed(2021)
            data_size = embed_feat_bank_cpu.shape[0]
            sample_idxs = np.random.choice(data_size, data_size//3, replace=False)
            embed_feat_bank_cpu = embed_feat_bank_cpu[sample_idxs, :]
        
        embed_feat_bank_cpu = TSNE(n_components=2, init="pca", random_state=0).fit_transform(embed_feat_bank_cpu)

        best_score = 0.0
        for coeff in coeff_list:
            Ct = max(int(args.class_num * coeff), 2)
            kmeans = KMeans(n_clusters=Ct, random_state=0).fit(embed_feat_bank_cpu)
            cluster_labels = kmeans.labels_
            sil_score = silhouette_score(embed_feat_bank_cpu, cluster_labels)
            
            if sil_score > best_score:
                best_score = sil_score
                best_coeff = coeff
        args.logger.info("Performing C_t estimation Done!")
        
    # args.logger.info("GLOBAL BEST COEFF: %s" % best_coeff)
    Ct = int(args.class_num * best_coeff)
    data_num = pred_cls_bank.shape[0]
    pos_topk_num = int(data_num / Ct)

    global known_space_basis
    global unknown_space_basis
    if epoch_idx == 0.0:
        src_cls_feat = model.class_layer.fc.weight.data #[C, D]
        u, s, vt = torch.linalg.svd(src_cls_feat.t())
        main_r = args.class_num
        known_space_basis = u[:, :main_r].t() #[C, D]
        known_space_basis = known_space_basis / torch.norm(known_space_basis, p=2, dim=-1, keepdim=True) #[C, D]
        unknown_space_basis = u[:, main_r:].t() #[D-C, D]
        unknown_space_basis = unknown_space_basis / torch.norm(unknown_space_basis, p=2, dim=-1, keepdim=True) #[D-C, D]
    
    known_proj_cords = torch.einsum("nd, cd -> nc", embed_feat_bank, known_space_basis) #[N, C]
    unknown_proj_cords = torch.einsum("nd, cd -> nc", embed_feat_bank, unknown_space_basis) #[N, D-C]
    
    known_proj_feat = torch.einsum("nc, cd -> nd", known_proj_cords, known_space_basis)
    unknwon_proj_feat = torch.einsum("nc, cd -> nd", unknown_proj_cords, unknown_space_basis)
    
    known_proj_norm = torch.norm(known_proj_cords, p=2, dim=-1) #[N]
    unknown_proj_norm = torch.norm(unknown_proj_cords, p=2, dim=-1) #[N]
    
    known_proj_norm_expand = known_proj_norm.unsqueeze(0).expand([args.class_num, -1]) #[C, N]
    unknown_proj_norm_expand = unknown_proj_norm.unsqueeze(0).expand([args.class_num, -1]) #[C, N]
    
    known_proj_feat_expand = known_proj_feat.unsqueeze(0).expand([args.class_num, -1, -1]) #[C, N, D]
    unknown_proj_feat_expand = unknwon_proj_feat.unsqueeze(0).expand([args.class_num, -1, -1]) #[C, N, D]
    
    unknown_space_norm_gm = GaussianMixture(n_components=2, random_state=0).fit(unknown_proj_norm.cpu().view(-1, 1))
    gaussian_two_mus = torch.tensor(unknown_space_norm_gm.means_).squeeze().cuda()

    gaussian_mu1 = torch.min(gaussian_two_mus)
    gaussian_mu2 = torch.max(gaussian_two_mus)
    # target prototype construction 
    embed_feat_bank_expand = embed_feat_bank.unsqueeze(0).expand([args.class_num, -1, -1]) #[C, N, D]
    sorted_pred_cls, sorted_pred_cls_idxs = torch.sort(pred_cls_bank, dim=0, descending=True)
    pos_topk_idxs = sorted_pred_cls_idxs[:pos_topk_num, :].t() #[C, pos_topk_num]
    pos_topk_idxs_feat_expand = pos_topk_idxs.unsqueeze(2).expand([-1, -1, args.embed_feat_dim]) #[C, pos_topk_num, D]
    pos_feat_sample = torch.gather(embed_feat_bank_expand, 1, pos_topk_idxs_feat_expand) #[C, pos_topk_num, D]
    
    tar_pos_feat_proto = torch.mean(pos_feat_sample, dim=1) #[C, D]
    tar_pos_feat_proto = tar_pos_feat_proto / torch.norm(tar_pos_feat_proto, p=2, dim=-1, keepdim=True) #[C, D]
    
    # source anchors construction 
    src_pos_feat_proto = model.class_layer.fc.weight.data / torch.norm(model.class_layer.fc.weight.data, p=2, dim=-1, keepdim=True) #[C, D]
    
    
    tar_psd_pos_feat_simi = torch.einsum("nd, cd -> nc", embed_feat_bank, tar_pos_feat_proto) #[N, C]
    tar_psd_pos_feat_simi = torch.clamp(tar_psd_pos_feat_simi, min=0.0)
    
    src_psd_pos_feat_simi = torch.einsum("nd, cd -> nc", embed_feat_bank, src_pos_feat_proto) #[N, C]
    src_psd_pos_feat_simi = torch.clamp(src_psd_pos_feat_simi, min=0.0)

    # per sample common score
    per_sample_fuse_common_score = torch.sqrt((1.0 - torch.exp(-tar_psd_pos_feat_simi)) * (torch.exp(src_psd_pos_feat_simi - 1.0)))

    # Instance-level decision boundaries.
    per_sample_per_cls_thresh = torch.zeros_like(pred_cls_bank) #[N, C]
    per_cls_norm_prior = torch.mean(torch.gather(unknown_proj_norm_expand, dim=1, index=pos_topk_idxs), dim=1,).unsqueeze(0) #[1, C]
    per_sample_per_cls_thresh = per_sample_per_cls_thresh + per_cls_norm_prior#[N, C]
    per_cls_thresh_gap = torch.clamp(gaussian_mu2 - per_cls_norm_prior, min=0.0) #[1, C]
    per_sample_per_cls_thresh = per_sample_per_cls_thresh + per_sample_fuse_common_score * per_cls_thresh_gap
    
    # Obtain psuedo-labels
    psd_label = torch.argmax(per_sample_fuse_common_score, dim=-1)
    psd_label_weight = torch.ones_like(psd_label).float()    
    psd_label_oh = psd_label.clone()
    for i in range(args.class_num):
        label_idxs = torch.where(psd_label == i)[0]
        
        alpha = 1e-4
        psd_label[label_idxs] = torch.where(unknown_proj_norm[label_idxs] >= per_sample_per_cls_thresh[label_idxs, i], args.class_num, psd_label[label_idxs])
        # Based on student's t distribution
        psd_label_weight[label_idxs] =  1.0 - torch.pow((1 + (unknown_proj_norm[label_idxs] - per_sample_per_cls_thresh[label_idxs, i])**2 / alpha), - (alpha + 1.) / 2.)
        
        unkown_idxs = unknown_proj_norm[label_idxs] >= gaussian_mu2
        known_idxs = unknown_proj_norm[label_idxs] < per_cls_norm_prior[0, i]
        psd_label_weight[label_idxs][unkown_idxs] = 1.0
        psd_label_weight[label_idxs][known_idxs] = 1.0
    
    psd_unknown_flg = (psd_label == args.class_num) #[N]
    psd_known_flg = (psd_label != args.class_num)
    
    psd_label_onehot = torch.zeros_like(pred_cls_bank).scatter_(1, psd_label_oh.unsqueeze(1), 1.0) #[N, C]
    psd_label_onehot[psd_unknown_flg, :] = 1.0
    psd_label_onehot = psd_label_onehot / (torch.sum(psd_label_onehot, dim=-1, keepdim=True) + 1e-5) #[N, C]
    psd_label_onehot = psd_label_onehot.cuda()
    
    per_class_num = np.zeros((len(class_list)))
    pre_class_num = np.zeros_like(per_class_num)
    per_class_correct = np.zeros_like(per_class_num)
    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_bank == label)[0]
        correct_idx = torch.where(psd_label[label_idx] == label)[0]
        pre_class_num[i] = float(len(torch.where(psd_label == label)[0]))
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))
    per_class_acc = per_class_correct / (per_class_num + 1e-5)
    
    args.logger.info("PSD AVG ACC:\t" + "{:.3f}".format(np.mean(per_class_acc)))
    args.logger.info("PSD PER ACC:\t" + "\t".join(["{:.3f}".format(item) for item in per_class_acc]))
    args.logger.info("PER CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_num]))
    args.logger.info("PRE CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in pre_class_num]))
    args.logger.info("PRE ACC NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_correct]))
    
    return psd_label_onehot, pred_cls_bank, embed_feat_bank, known_space_basis, unknown_space_basis, psd_unknown_flg, psd_label_weight
    
known_space_basis = None
unknown_space_basis = None
def train(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
    
    model.eval()
    psd_label_onehot_bank, pred_cls_bank, embed_feat_bank,\
    known_space_basis, unknown_space_basis, psd_unknown_bank, psd_label_weight_bank = obtain_LEAD_pseudo_labels(args, model, test_dataloader, epoch_idx=epoch_idx)
    model.train()

    local_KNN = 4
    all_pred_loss_stack = []
    psd_pred_loss_stack = []
    knn_pred_loss_stack = []
    reg_pred_loss_stack = []

    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)
    for imgs_train, _, imgs_label, imgs_idx in tqdm(train_dataloader, ncols=60):
        
        iter_idx += 1
        imgs_idx = imgs_idx.cuda()
        imgs_train = imgs_train.cuda()
        
        psd_label = psd_label_onehot_bank[imgs_idx] #[B, C]
        psd_weight = psd_label_weight_bank[imgs_idx].unsqueeze(1) #[B, 1]
        
        embed_feat, pred_cls = model(imgs_train, apply_softmax=True)
        
        # mathcal{L}_{ce}
        psd_pred_loss = torch.sum(-psd_label * psd_weight * torch.log(pred_cls + 1e-5), dim=-1).mean()
        
        # mathcal{L}_{reg}
        embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=-1, keepdim=True)
        knfeat_proj_cords = torch.einsum("nd, cd -> nc", embed_feat, known_space_basis) #[B, D]
        knfeat_proj_norms = torch.norm(knfeat_proj_cords, p=2, dim=-1, keepdim=True) #[B, 1]
        
        unfeat_proj_cords = torch.einsum("nd, cd -> nc", embed_feat, unknown_space_basis) #[B, D]
        unfeat_proj_norms = torch.norm(unfeat_proj_cords, p=2, dim=-1, keepdim=True) #[B, 1]
        
        feat_proj_norms = torch.cat([knfeat_proj_norms, unfeat_proj_norms], dim=-1) #[B, 2]
        feat_proj_probs = torch.softmax(feat_proj_norms, dim=-1)

        psd_unknown_flg = psd_unknown_bank[imgs_idx].long() #[B]
        psd_unknown_flg = torch.zeros_like(feat_proj_norms).scatter_(1, psd_unknown_flg.unsqueeze(1), 1.0)
        feat_reg_loss = torch.sum(-psd_unknown_flg * psd_weight * torch.log(feat_proj_probs + 1e-5), dim=-1).mean()
        
        # mathcal{L}_{con}
        with torch.no_grad():
            feat_dist = torch.einsum("bd, nd -> bn", embed_feat, embed_feat_bank) #[B, N]
            nn_feat_idx = torch.topk(feat_dist, k=local_KNN+1, dim=-1, largest=True)[-1] #[B, local_KNN+1]
            nn_feat_idx_0 = nn_feat_idx[:, 1:] #[B, local_KNN]
            nn_pred_cls = torch.mean(pred_cls_bank[nn_feat_idx_0], dim=1) #[B, C]
            # update the pred_cls and embed_feat bank 
            pred_cls_bank[imgs_idx] = pred_cls
            embed_feat_bank[imgs_idx] = embed_feat
            
        knn_pred_loss = torch.sum( -nn_pred_cls * torch.log(pred_cls + 1e-5), dim=-1).mean()
        
 
        loss = args.lam_psd * psd_pred_loss + feat_reg_loss + knn_pred_loss    
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_pred_loss_stack.append(loss.cpu().item())
        psd_pred_loss_stack.append(psd_pred_loss.cpu().item())
        knn_pred_loss_stack.append(knn_pred_loss.cpu().item())
        reg_pred_loss_stack.append(feat_reg_loss.cpu().item())
        
    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)
    train_loss_dict["psd_pred_loss"] = np.mean(psd_pred_loss_stack)
    train_loss_dict["con_pred_loss"] = np.mean(knn_pred_loss_stack)
    train_loss_dict["reg_pred_loss"] = np.mean(reg_pred_loss_stack)
    
    return train_loss_dict

@torch.no_grad()
def test(args, model, dataloader, src_flg=False):
    
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []
    
    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda() 
        _, pred_cls = model(imgs_test, apply_softmax=True)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())
    
    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]

    h_score, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=args.w_0)
    return h_score, known_acc, unknown_acc
    
def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    
    model = SFUniDA(args)
    
    model = model.cuda()

    if args.note is None:
        save_dir = os.path.join(this_dir, "checkpoints", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                                args.target_label_type,
                                "{}_psd_{}".format(args.source_train_type, args.lam_psd))
    else:
        save_dir = os.path.join(this_dir, "checkpoints", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                                args.target_label_type,
                                "{}_psd_{}_{}".format(args.source_train_type, args.lam_psd, args.note))
        
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_target_training.txt")
    
    if args.reset:
        raise ValueError
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")
    
    shutil.copy("./train_target.py", os.path.join(args.save_dir, "train_target.py"))
    shutil.copy("./utils/net_utils.py", os.path.join(args.save_dir, "net_utils.py"))
    
    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    for k, v in model.class_layer.named_parameters():
        v.requires_grad = False  
        
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    
    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size*2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)
    
    notation_str =  "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="
    
    args.logger.info(notation_str)

    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0
    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        # Train on target
        loss_dict =train(args, model, target_train_dataloader, target_test_dataloader, optimizer, epoch_idx)
        args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f},\n\
                          train_psd_loss:{:.3f}, train_reg_loss:{:.3f}, train_con_loss:{:.3f},".format(epoch_idx+1, args.epochs,
                                        loss_dict["all_pred_loss"], loss_dict["psd_pred_loss"], loss_dict["reg_pred_loss"], loss_dict["con_pred_loss"],))
        
        # Evaluate on target
        hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)
        args.logger.info("Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))
        
        if args.target_label_type == 'PDA' or args.target_label_type == 'CLDA':
            if knownacc >= best_known_acc:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx
                
                # checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)         
                # torch.save({
                #     "epoch":epoch_idx,
                #     "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
        else:
            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx
            
                # checkpoint_file = "{}_SFDA_best_target_checkpoint.pth".format(args.dataset)         
                # torch.save({
                #     "epoch":epoch_idx,
                #     "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
            
        args.logger.info("Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_h_score, best_known_acc, best_unknown_acc))
            
if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    
    args.checkpoint = os.path.join("checkpoints", args.dataset, "source_{}".format(args.s_idx),\
                    "source_{}_{}".format(args.source_train_type, args.target_label_type),
                    "latest_source_checkpoint.pth")
    args.reset = False
    main(args)