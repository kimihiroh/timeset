#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_dev=./data/brat/dev/
dirpath_test=./data/brat/test/
dirpath_output=./data/preprocess/timeset/
dirpath_log=./log/

python src/preprocess_timeset.py \
    --dirpath_dev $dirpath_dev \
    --dirpath_test $dirpath_test \
    --dirpath_output $dirpath_output \
    --dirpath_log $dirpath_log \
