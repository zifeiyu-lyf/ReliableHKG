#!/bin/bash

cd ../code/
python train.py --dataset_name FB-25 --setting Inductive --lr 1e-3 --dim 32 \
                --num_epoch 390 --val_dur 390 --num_init_layer 3 --num_layer 5 --num_head 8 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.6 --model_dropout 0.1 --smoothing 0.0 \
                --log_name FB-25 --exp ReliableHKG --msg_add_tr
