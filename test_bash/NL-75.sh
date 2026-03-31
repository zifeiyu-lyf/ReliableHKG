#!/bin/bash

cd ../code/
python test.py --log_name NL-75 --exp ReliableHKG --dataset_name NL-75 --test_epoch 660 --setting Inductive \
                --dim 32 --num_init_layer 4 --num_layer 1 --num_head 2 --msg_add_tr  --enable_conformal --stratified
