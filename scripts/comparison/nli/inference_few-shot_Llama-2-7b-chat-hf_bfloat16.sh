#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0

eval "$(conda shell.bash hook)"
$CONDA activate timeset

dirpath_output=./output/comparison/
dirpath_output_score=./output_score/comparison/
dirpath_log=./log
filepath_test=./data/preprocessed/timeset-sample/test.json
filepath_dev=./data/preprocessed/timeset-sample/dev.json

dataset_name=ctf-nli
inference_type=few-shot
seeds=( 7 )

batch_size=8
max_new_tokens=512
num_demonstrations=( 0 1 2 3 )
temperature=0
model_id=meta-llama/Llama-2-7b-chat-hf
precision_type=bfloat16
markers=( eid )
representations=( mention )
num_gpu=1

nvidia-smi

for seed in "${seeds[@]}"; do
    for marker in "${markers[@]}"; do
        for representation in "${representations[@]}"; do
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
                    --marker "$marker" \
                    --model_id $model_id \
                    --num_demonstration "$num_demonstration" \
                    --num_gpu $num_gpu \
                    --num_cpu 4 \
                    --precision_type $precision_type \
                    --representation "$representation" \
                    --seed "$seed" \
                    --temperature $temperature \
                    --dirpath_model_cache "$HF_HOME/hub/"
            done
        done
    done
done
