#!/bin/bash

PCB_tri_path=$1


CUDA_VISIBLE_DEVICES=$2 python  PCB_tri_partial_column.py -d market -a resnet50_pseudo_column -b 64 -j 1 --epochs 60 --log ${PCB_tri_path} --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/datasets/Market-1501/  --step-size 80  --lr 0.1 --num_parts $3  --epochs 130  --num-instances 4 #--evaluate

CUDA_VISIBLE_DEVICES=$2 python  PCB_tri_partial_column.py -d market -a resnet50_pseudo_column -b 64 -j 1 --epochs 60 --log ${PCB_tri_path} --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/datasets/Market-1501/  --step-size 70  --lr 0.1 --num_parts $3  --epochs 160  --resume ${PCB_tri_path}/checkpoint_130.pth.tar --num-instances 4 #--evaluate

cp reid/utils/data/transforms.py ${PCB_tri_path}/
cp reid/trainers_tri_pseudo_column.py ${PCB_tri_path}/
cp reid/models/resnet_pseudo_column.py ${PCB_tri_path}/

#====================test PCB_tri==================#
