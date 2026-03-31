#!/bin/bash

cd ../code/
python train.py --dataset_name WK-75 --setting Inductive --lr 5e-4 --dim 32 \
                --num_epoch 250 --val_dur 250 --num_init_layer 4 --num_layer 5 --num_head 4 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.5 --model_dropout 0.05 --smoothing 0.0 \
                --log_name WK-75 --exp ReliableHKG --msg_add_tr
