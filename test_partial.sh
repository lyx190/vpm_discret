CUDA_VISIBLE_DEVICES=$1 python -m pdb PCB_partial_test.py -d market -a resnet50_pseudo_column -b 32 -j 1  --log logs/market-1501/tmp --combine-trainval --feature 256 --height 384 --width 128  --data-dir ~/datasets/Market-1501/  --step-size 60  --resume $2 --evaluate --ratio $3 --num_parts $4


#CUDA_VISIBLE_DEVICES=8,9 python Softmax_Triplet_Partial.py -a resnet50 -b 128 -d market -j4 --height 256 --width 128 --features 256 --combine-trainval --num-instances 8 --log logs/tmp  --data-dir ~/dataset/Market-1501/ --margin 1 --ratio 0.8 --resume ~/backup_32server/open-reid/logs/market-1501/soft_triplet_partial0.5/checkpoint.pth.tar --evaluate --ratio $1
