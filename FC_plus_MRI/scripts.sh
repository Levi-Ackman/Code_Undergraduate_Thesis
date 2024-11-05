#!/bin/bash

mkdir -p ./logs

export CUDA_VISIBLE_DEVICES=1
model_name=CAA
log_dir="./logs/"               
cmd="python -u run.py \
    --model $model_name >${log_dir}$.log"
    eval $cmd
