#!/bin/bash

cd ../code/
python test.py --log_name NL-50 --exp ReliableHKG --dataset_name NL-50 --test_epoch 390 --setting Inductive \
                --dim 32 --num_init_layer 5 --num_layer 8 --num_head 4 --msg_add_tr  --enable_conformal --stratified