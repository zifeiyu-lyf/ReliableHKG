#!/bin/bash

cd ../code/
python test.py --log_name FB-75 --exp ReliableHKG --dataset_name FB-75 --test_epoch 140 --setting Inductive \
                --dim 32 --num_init_layer 3 --num_layer 5 --num_head 8 --msg_add_tr --enable_conformal --stratified


