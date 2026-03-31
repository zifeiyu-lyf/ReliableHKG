#!/bin/bash

cd ../code/
python train.py --dataset_name FB-100 --setting Inductive --lr 1.5e-4 --dim 32 \
                --num_epoch 310 --val_dur 310 --num_init_layer 4 --num_layer 5 --num_head 4 \
                --early_stop 0 --batch_num 30 --train_graph_ratio 0.3 --model_dropout 0.1 --smoothing 0.0 \
                --log_name FB-100 --exp ReliableHKG
