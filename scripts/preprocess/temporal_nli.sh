#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_input=./data/temporal_nli/data/recast/tempeval3/
dirpath_output=./data/preprocess/temporal_nli/
dirpath_log=./log/

python src/preprocess_temporal_nli.py \
    --dirpath_input $dirpath_input \
    --dirpath_output $dirpath_output \
    --dirpath_log $dirpath_log \
