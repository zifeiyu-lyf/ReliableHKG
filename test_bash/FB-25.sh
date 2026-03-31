#!/bin/bash

cd ../code/
python test.py --log_name FB-25 --exp ReliableHKG --dataset_name FB-25 --test_epoch 390 --setting Inductive \
                --dim 32 --num_init_layer 3 --num_layer 5 --num_head 8 --msg_add_tr --enable_conformal --stratified
