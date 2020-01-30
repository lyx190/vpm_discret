#!/bin/bash
log='../logs'
resume_path='/mnt/trained_model/logs_280120/checkpoint_130.pth.tar'
ratio=1.0

python3 PCB_partial_test.py -d market -a resnet50_pseudo_column -b 32 -j 1  --log ${log} --combine-trainval --feature 256 --height 384 --width 128  --data-dir /mnt/datasets/Market/  --step-size 60  --resume ${resume_path} --evaluate --ratio ${ratio} --num_parts 6

