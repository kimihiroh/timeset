#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0

conda activate timeset

dirpath_output=# TODO: dir for output files

dirpath_log=./log
filepath_data_train=data/preprocess/matres/train.json
filepath_data_dev=data/preprocess/matres/dev.json

dataset_name=matres

batch_sizes=( 16 32 )
model_id=microsoft/deberta-v3-large
num_epoch=10
num_gpu=1
precision_type=bfloat16
finetune_type=ft

learning_rates=( 1e-4 1e-5 1e-6 )

for batch_size in "${batch_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do

        python src/finetune_classification.py \
            --batch_size "$batch_size" \
            --dataset_name $dataset_name \
            --dirpath_log $dirpath_log \
            --dirpath_output "$dirpath_output" \
            --filepath_data_train $filepath_data_train \
            --filepath_data_dev $filepath_data_dev \
            --finetune_type $finetune_type \
            --learning_rate "$learning_rate" \
            --model_id $model_id \
            --num_epoch $num_epoch \
            --num_gpu $num_gpu \
            --precision_type $precision_type \
            --save_model_weight \
            --warmup

    done
done
