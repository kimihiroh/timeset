#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_output=./scripts
task=comparison
# task=benchmark
dirpath_log=./log

python src/generate_bash_scripts_for_experiments.py \
    --dirpath_log $dirpath_log \
    --task $task \
    --dirpath_output $dirpath_output \
