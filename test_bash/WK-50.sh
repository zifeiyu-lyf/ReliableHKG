#!/bin/bash

cd ../code/
python test.py --log_name WK-50 --exp ReliableHKG --dataset_name WK-50 --test_epoch 230 --setting Inductive \
                --dim 32 --num_init_layer 3 --num_layer 5 --num_head 8 --msg_add_tr --enable_conformal --stratified