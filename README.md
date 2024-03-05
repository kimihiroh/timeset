# TimeSET: Formulation Comparison for Timeline Construction using LLMs

This repository contains the data and code for the paper, ["Formulation Comparison for Timeline Construction using LLMs" (Hasegawa et al., arXiv 2024)](https://arxiv.org/submit/5426384/view).

## TimeSET
* `brat` format: `data/brat`
* `json` format: `data/preprocess/timeset`
    * Data used in the paper: `data/preprocess/timeset-sample/`
Check [data](https://github.com/kimihiroh/timeset/blob/main/data/) for more details.

## Environment Setup

```bash
conda create -y -n timeset python=3.10
conda activate timeset
conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pre-commit install
```
Then, set `HF_HOME`, e.g.,
```bash
export HF_HOME="<some dir>/.cache/huggingface/"
```

## Formulation Comparison
### Download model weights
```bash
bash scripts/download.sh
```
### Experiment
```bash
bash scripts/comparison/{formulation}/{model}.sh
```
* `output` contains the actual outputs from models.
* `output_score` contains the evaluation results/scores.

## Benchmarking
### Download data
```bash
cd data
# TORQUE (annotation+document)
git clone git@github.com:qiangning/TORQUE-dataset.git
# Temporal NLI
git clone git@github.com:sidsvash26/temporal_nli.git
# MATRES (annotation)
git clone git@github.com:qiangning/MATRES.git
# MATRES (document)
git clone git@github.com:qiangning/StructTempRel-EMNLP17.git
# TDDiscourse
git clone git@github.com:aakanksha19/TDDiscourse.git
# [Additional]
git@github.com:Jryao/temporal_dependency_graphs_crowdsourcing.git
```
### Preprocess
```bash
# TORQUE
bash scripts/preprocess/torque.sh
# MATRES
bash scripts/preprocess/matres.sh
# temporal nli (TempEval3)
# TODO: add customized preprocessing script & instruction
bash scripts/preprocess/temporal_nli.sh
# TDDiscourse (TDDMan)
bash scripts/preprocess/tddiscourse.sh
# ctf
bash scripts/preprocess/ctf.sh
```
### Download model weights
```bash
bash scripts/download.sh
```
### Finetuning
```bash
# {method} = [ft, peft]
bash scripts/benchmark/{dataset}/{method}_{model}.sh
```
### Inference
```bash
bash scripts/benchmark/{dataset}/inference_{method}_*.sh
```
* `output` contains the actual outputs from models.
* `output_score` contains the evaluation results/scores.

## Citation
If you find this work helpful in your research, please consider citing our work,
```bib
TBU
```

## Issues
For any issues, questions or requests, please create a [Github Issue](https://github.com/kimihiroh/timeset/issues).
