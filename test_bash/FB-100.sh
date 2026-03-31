#!/bin/bash

cd ../code/
python test.py --log_name FB-100 --exp ReliableHKG --dataset_name FB-100 --test_epoch 310 --setting Inductive \
                --dim 32 --num_init_layer 4 --num_layer 5 --num_head 4  --enable_conformal --stratified
