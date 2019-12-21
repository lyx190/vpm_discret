#!/bin/bash

PCB_tri_path='~/logs'

python3 -W ignore PCB_tri_partial_column.py -d market -a resnet50_pseudo_column -b 64 -j 1 --epochs 60 --log ${PCB_tri_path} --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir ~/Market/  --step-size 80  --lr 0.1 --num_parts 6  --epochs 130  #--evaluate

cp reid/utils/data/transforms.py ${PCB_tri_path}/
cp reid/trainers_tri_pseudo_column.py ${PCB_tri_path}/
cp reid/models/resnet_pseudo_column.py ${PCB_tri_path}/

#====================test PCB_tri==================#
