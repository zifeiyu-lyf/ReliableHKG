#!/bin/bash

cd ../code/
python test.py --log_name NL-25 --exp ReliableHKG --dataset_name NL-25 --test_epoch 260 --setting Inductive \
                --dim 32 --num_init_layer 2 --num_layer 10 --num_head 2 --msg_add_tr  --enable_conformal --stratified
