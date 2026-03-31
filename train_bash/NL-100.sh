#!/bin/bash

cd ../code/
python train.py --dataset_name NL-100 --setting Inductive --lr 2e-4 --dim 32 \
                --num_epoch 160 --val_dur 160 --num_init_layer 4 --num_layer 4 --num_head 4 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.35 --model_dropout 0.05 --smoothing 0.0 \
                --log_name NL-100 --exp ReliableHKG
