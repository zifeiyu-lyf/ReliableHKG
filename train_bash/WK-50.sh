#!/bin/bash

cd ../code/
python train.py --dataset_name WK-50 --setting Inductive --lr 2e-4 --dim 32 \
                --num_epoch 230 --val_dur 230 --num_init_layer 3 --num_layer 5 --num_head 8 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.3 --model_dropout 0.1 --smoothing 0.0 \
                --log_name WK-50 --exp ReliableHKG --msg_add_tr
