#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_input=./data/TORQUE-dataset/data/
dirpath_output=./data/preprocess/torque/
dirpath_log=./log/

python src/preprocess_torque_original.py \
    --dirpath_input $dirpath_input \
    --dirpath_output $dirpath_output \
    --dirpath_log $dirpath_log \
