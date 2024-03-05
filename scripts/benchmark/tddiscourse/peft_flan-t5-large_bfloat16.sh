#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"

conda activate timeset

dirpath_output=# TODO: dir for output files

dirpath_log=./log
filepath_data_train=data/preprocess/tddiscourse/train.json
filepath_data_dev=data/preprocess/tddiscourse/dev.json

dataset_name=tddiscourse

batch_size=8
model_id=google/flan-t5-large
num_epoch=10
precision_type=bfloat16
finetune_type=peft
peft_type=lora
lora_dimensions=( 16 64 )
lora_alphas=( 16 64 )
lora_dropout=0.1

learning_rates=( 1e-4 1e-5 1e-6 )

for lora_alpha in "${lora_alphas[@]}"; do
    for lora_dimension in "${lora_dimensions[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do

            python src/finetune_generation.py \
                --batch_size $batch_size \
                --dataset_name $dataset_name \
                --dirpath_log $dirpath_log \
                --dirpath_output "$dirpath_output" \
                --filepath_data_train $filepath_data_train \
                --filepath_data_dev $filepath_data_dev \
                --finetune_type $finetune_type \
                --learning_rate "$learning_rate" \
                --lora_alpha "$lora_alpha" \
                --lora_dimension "$lora_dimension" \
                --lora_dropout $lora_dropout \
                --model_id $model_id \
                --num_epoch $num_epoch \
                --peft_type $peft_type \
                --precision_type $precision_type \
                --save_model_weight \
                --warmup
        done
    done
done
