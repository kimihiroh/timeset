# Preparation
## Data
Check [data](https://github.com/kimihiroh/timeset/blob/main/data#benchmark) for more details.

## Download model weights
```bash
bash scripts/download.sh
```
## Generate scripts
```bash
# 1. modify src/template_script/benchmark_*.txt
# 2. generate scripts after setting `task=benchmark`
bash scripts/generate_bash_scripts_for_experiments.sh
```
# Experiment
## Finetuning
```bash
# {method} = [ft, peft]
bash scripts/benchmark/{dataset}/{method}_{model}.sh
```
Check `notebooks/check-ft-results.ipynb` for finetuning results.
## Inference
```bash
bash scripts/benchmark/{dataset}/inference_{method}_*.sh
```
* `output` contains the actual outputs from models.
* `output_score` contains the evaluation results/scores.
## Result
Check `notebooks/check-benchmark-result.ipynb` for results.
