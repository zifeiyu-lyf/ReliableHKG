#!/bin/bash

cd ../code/
python test.py --log_name NL-100 --exp ReliableHKG --dataset_name NL-100 --test_epoch 160 --setting Inductive \
                --dim 32 --num_init_layer 4 --num_layer 4 --num_head 4  --enable_conformal --stratified
