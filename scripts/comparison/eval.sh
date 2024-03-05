#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_output=./output/
dirpath_log=./log
filepath_test=./data/preprocess/timeset-sample/test.json

filenames_pred=(
    # add filenames
)

for filename_pred in "${filenames_pred[@]}"; do

    python src/evaluate.py \
        --dirpath_log $dirpath_log \
        --filepath_test $filepath_test \
        --filepath_pred $dirpath_output/"$filename_pred"

done
