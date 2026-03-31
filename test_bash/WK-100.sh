#!/bin/bash

cd ../code/
python test.py --log_name WK-100 --exp ReliableHKG --dataset_name WK-100 --test_epoch 410 --setting Inductive \
                --dim 32 --num_init_layer 3 --num_layer 5 --num_head 2 --enable_conformal --stratified
