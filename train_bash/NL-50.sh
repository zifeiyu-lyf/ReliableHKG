#!/bin/bash

cd ../code/
python train.py --dataset_name NL-50 --setting Inductive --lr 1e-3 --dim 32 \
                --num_epoch 390 --val_dur 390 --num_init_layer 5 --num_layer 8 --num_head 4 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.4 --model_dropout 0.1 --smoothing 0.0 \
                --log_name NL-50 --exp ReliableHKG --msg_add_tr
