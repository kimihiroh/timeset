#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0
eval "$(conda shell.bash hook)"

conda activate timeset

dirpath_output=./output/benchmark/
dirpath_output_score=./output_score/benchmark/
dirpath_log=./log
filepath_test=data/preprocessed/matres/test.json
filepath_dev=data/preprocessed/matres/dev.json

dataset_name=matres
inference_type=few-shot
seeds=( 7 )

batch_size=8
max_new_tokens=64
num_demonstrations=( 0 1 3 5 10 )
temperature=0
model_id=meta-llama/Llama-2-7b-chat-hf
precision_type=bfloat16
num_gpu=1

for seed in "${seeds[@]}"; do
    for num_demonstration in "${num_demonstrations[@]}"; do

        python src/inference_generation_vllm.py \
            --batch_size $batch_size \
            --dataset_name $dataset_name \
            --dirpath_log $dirpath_log \
            --dirpath_output $dirpath_output \
            --dirpath_output_score $dirpath_output_score \
            --filepath_test $filepath_test \
            --filepath_dev $filepath_dev \
            --inference_type $inference_type \
            --max_new_tokens $max_new_tokens \
            --model_id $model_id \
            --num_demonstration "$num_demonstration" \
            --num_gpu $num_gpu \
            --precision_type $precision_type \
            --seed "$seed" \
            --temperature $temperature \
            --dirpath_model_cache "$TRANSFORMERS_CACHE"
    done
done
