#!/bin/bash

cd ../code/
python train.py --dataset_name NL-25 --setting Inductive --lr 2e-3 --dim 32 \
                --num_epoch 260 --val_dur 260 --num_init_layer 2 --num_layer 10 --num_head 2 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.5 --model_dropout 0.05 --smoothing 0.0 \
                --log_name NL-25 --exp ReliableHKG --msg_add_tr
