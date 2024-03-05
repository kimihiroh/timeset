#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_input=./data/preprocess/timeset/
dirpath_output=./data/preprocess/timeset-metadata/

python src/analyze_timeset.py \
    --dirpath_input $dirpath_input \
    --dirpath_output $dirpath_output
