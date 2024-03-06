gpuid=${1:-0}
random_seed=${2:-2021}

# export CUDA_VISIBLE_DEVICES=$gpuid

echo "PDA SOURCE TRAIN ON OFFICE"
python train_source.py  --dataset Office --s_idx 0  --target_label_type PDA --epochs 50 --lr 0.01 
python train_source.py  --dataset Office --s_idx 1  --target_label_type PDA --epochs 50 --lr 0.01 
python train_source.py  --dataset Office --s_idx 2  --target_label_type PDA --epochs 50 --lr 0.01 

echo "PDA SOURCE TRAIN ON OFFICEHOME"
python train_source.py  --dataset OfficeHome --s_idx 0  --target_label_type PDA --epochs 50 --lr 0.01 
python train_source.py  --dataset OfficeHome --s_idx 1  --target_label_type PDA --epochs 50 --lr 0.01 
python train_source.py  --dataset OfficeHome --s_idx 2  --target_label_type PDA --epochs 50 --lr 0.01 
python train_source.py  --dataset OfficeHome --s_idx 3  --target_label_type PDA --epochs 50 --lr 0.01 

echo "PDA SOURCE TRAIN ON VisDA"
python train_source.py --backbone_arch resnet50 --dataset VisDA --s_idx 0  --target_label_type PDA --epochs 10 --lr 0.001 