#!/bin/bash

cd ../code/
python train.py --dataset_name NL-75 --setting Inductive --lr 2e-4 --dim 32 \
                --num_epoch 660 --val_dur 660 --num_init_layer 4 --num_layer 1 --num_head 2 \
                --early_stop 0 --batch_num 35 --train_graph_ratio 0.15 --model_dropout 0.0 --smoothing 0.0 \
                --log_name NL-75 --exp ReliableHKG --msg_add_tr
