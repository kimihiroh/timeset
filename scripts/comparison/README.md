# Preparation
## Download model weights
```bash
bash scripts/download.sh
```
## Generate scripts
```bash
# 1. modify src/template_script/comparison_*.txt
# 2. generate scripts after setting `task=comparison`
bash scripts/generate_bash_scripts_for_experiments.sh
```
# Experiment
```bash
# formulation = {mrc, mrc-cot, nli, pairwise, timeline, timeline-code, timeline-cot}
bash scripts/comparison/{formulation}/{model}.sh
```
* `output` contains the actual outputs from models.
* `output_score` contains the evaluation results/scores.
## Result
Check the following notebooks for illustration:
* `notebooks/result-formulation-comparison.ipynb`
* `notebooks/analysis-metadata-formulation-comparison.ipynb`
