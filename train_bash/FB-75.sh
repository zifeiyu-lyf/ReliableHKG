#!/bin/bash

cd ../code/
python train.py --dataset_name FB-75 --setting Inductive --lr 5e-4 --dim 32 \
                --num_epoch 140 --val_dur 140 --num_init_layer 3 --num_layer 5 --num_head 8 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.3 --model_dropout 0.1 --smoothing 0.0 \
                --log_name FB-75 --exp ReliableHKG --msg_add_tr
