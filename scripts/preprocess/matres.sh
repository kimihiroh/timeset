#!/usr/bin/bash

eval "$(conda shell.bash hook)"
conda activate timeset

dirpath_timebank=./data/StructTempRel-EMNLP17/data/TempEval3/Training/TBAQ-cleaned/TimeBank/
dirpath_aquaint=./data/StructTempRel-EMNLP17/data/TempEval3/Training/TBAQ-cleaned/AQUAINT/
dirpath_platinum=./data/StructTempRel-EMNLP17/data/TempEval3/Evaluation/te3-platinum/
dirpath_annotation=./data/MATRES/
dirpath_output=./data/preprocess/matres/
dirpath_log=./log/


python src/preprocess_matres.py \
    --dirpath_timebank $dirpath_timebank \
    --dirpath_aquaint $dirpath_aquaint \
    --dirpath_platinum $dirpath_platinum \
    --dirpath_annotation $dirpath_annotation \
    --dirpath_output $dirpath_output \
    --dirpath_log $dirpath_log \
