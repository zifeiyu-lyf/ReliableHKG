#!/bin/bash

cd ../code/
python test.py --log_name WK-75 --exp ReliableHKG --dataset_name WK-75 --test_epoch 250 --setting Inductive \
                --dim 32 --num_init_layer 4 --num_layer 5 --num_head 4 --msg_add_tr --enable_conformal --stratified