#!/bin/bash

cd ../code/
python train.py --dataset_name WK-100 --setting Inductive --lr 5e-4 --dim 32 \
                --num_epoch 410 --val_dur 410 --num_init_layer 3 --num_layer 5 --num_head 2 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.3 --model_dropout 0.05 --smoothing 0.0 \
                --log_name WK-100 --exp ReliableHKG
