gpuid=${1:-0}
random_seed=${2:-2021}

# export CUDA_VISIBLE_DEVICES=$gpuid

lam_psd=0.30
echo "OPDA Adaptation ON Office"
python train_target.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA

lam_psd=1.00
echo "OPDA Adaptation ON VisDA"
python train_target.py --backbone_arch resnet50 --lr 0.0001 --dataset VisDA --lam_psd $lam_psd --target_label_type OPDA  --epochs 30

lam_psd=2.00
echo "OPDA Adaptation ON Office-Home"
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA
python train_target.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001  --lam_psd $lam_psd --target_label_type OPDA

echo "OPDA Adaptation ON DomainNet"
python train_target.py --dataset DomainNet --s_idx 0 --t_idx 1 --lr 0.0001  --lam_psd $lam_psd --target_label_type OPDA --epochs 10
python train_target.py --dataset DomainNet --s_idx 0 --t_idx 2 --lr 0.0001  --lam_psd $lam_psd --target_label_type OPDA --epochs 10
python train_target.py --dataset DomainNet --s_idx 1 --t_idx 0 --lr 0.0001  --lam_psd $lam_psd --target_label_type OPDA --epochs 10
python train_target.py --dataset DomainNet --s_idx 1 --t_idx 2 --lr 0.0001  --lam_psd $lam_psd --target_label_type OPDA --epochs 10
python train_target.py --dataset DomainNet --s_idx 2 --t_idx 0 --lr 0.0001  --lam_psd $lam_psd --target_label_type OPDA --epochs 10
python train_target.py --dataset DomainNet --s_idx 2 --t_idx 1 --lr 0.0001  --lam_psd $lam_psd --target_label_type OPDA --epochs 10
