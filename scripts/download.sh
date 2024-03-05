#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_log=./log
num_gpu=4

model_ids=(
    t5-large
    google/flan-t5-large
    t5-3b
    google/flan-t5-xl
    meta-llama/Llama-2-7b-hf
    meta-llama/Llama-2-7b-chat-hf
    codellama/CodeLlama-7b-hf
    t5-11b
    google/flan-t5-xxl
    meta-llama/Llama-2-13b-hf
    meta-llama/Llama-2-13b-chat-hf
    codellama/CodeLlama-13b-hf
    codellama/CodeLlama-34b-hf
    meta-llama/Llama-2-70b-hf
    meta-llama/Llama-2-70b-chat-hf
)

for model_id in "${model_ids[@]}"; do

    nvidia-smi

    python src/download.py \
        --dirpath_log $dirpath_log \
        --model_id "$model_id" \
        --num_gpu $num_gpu

    nvidia-smi

    python src/download_vllm.py \
        --dirpath_log $dirpath_log \
        --dirpath_model_cache "$HF_CACHE/hub/" \
        --model_id "$model_id" \
        --num_gpu $num_gpu

done
