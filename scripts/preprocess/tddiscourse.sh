#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_annotation=./data/TDDiscourse/TDDMan/
dirpath_timebank=./data/StructTempRel-EMNLP17/data/TempEval3/Training/TBAQ-cleaned/TimeBank/
dirpath_output=./data/preprocess/tddiscourse/
dirpath_log=./log/

python src/preprocess_tddiscourse.py \
    --dirpath_annotation $dirpath_annotation \
    --dirpath_timebank $dirpath_timebank \
    --dirpath_output $dirpath_output \
    --dirpath_log $dirpath_log \
