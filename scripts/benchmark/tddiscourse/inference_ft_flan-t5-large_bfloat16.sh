#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"

conda activate timeset

dirpath_output=./output/benchmark/
dirpath_output_score=./output_score/benchmark/
dirpath_log=./log
filepath_test=data/preprocess/tddiscourse/test.json

dataset_name=tddiscourse
inference_type=ft
seeds=( 7 )

batch_size=8
model_id=google/flan-t5-large
precision_type=bfloat16
num_gpu=1

max_new_tokens=64
num_demonstration=0
temperature=0

dirpath_model=/usr1/datasets/kimihiro/llm-for-event-temporal-ordering/models/tddiscourse/flan-t5-large_bfloat16_ft_generation/seed7_bs16_lr0.0001_dimNone_alphaNone_dropNone
peft_model_path=None

for seed in "${seeds[@]}"; do
    python src/inference_generation.py \
        --batch_size $batch_size \
        --dataset_name $dataset_name \
        --dirpath_log $dirpath_log \
        --dirpath_model $dirpath_model \
        --dirpath_output $dirpath_output \
        --dirpath_output_score $dirpath_output_score \
        --filepath_test $filepath_test \
        --inference_type $inference_type \
        --model_id $model_id \
        --num_gpu $num_gpu \
        --precision_type $precision_type \
        --peft_model_path $peft_model_path \
        --max_new_tokens $max_new_tokens \
        --num_demonstration $num_demonstration \
        --temperature $temperature \
        --seed "$seed"
done
