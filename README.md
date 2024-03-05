# TimeSET: Formulation Comparison for Timeline Construction using LLMs

This repository contains the data and code for the paper: <br>
["Formulation Comparison for Timeline Construction using LLMs" (Hasegawa et al., arXiv 2024)](https://arxiv.org/abs/2403.00990).

## News
* 2024/03/04: TimeSET now includes 118 annotated documents (dev: 18, test: 100), over twice the number reported in the paper (dev: 10, test: 40).

## TimeSET
TimeSET is an evaluation dataset for timeline construction from text, consisting of diverse Wikinews articles.
It features two unique characteristics:
* saliency-based event selection
* partial-ordering annotation

Check [the paper](https://arxiv.org/abs/2403.00990) for more details.
### Data
* `brat` format: `data/brat`
* `json` format: `data/preprocess/timeset`
    * Original data used in the paper: `data/preprocess/timeset-sample/`

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
![Overview](https://github.com/kimihiroh/timeset/blob/main/notebooks/figures/overview_w_timeline.png)

Check [`scripts/comparison`](https://github.com/kimihiroh/timeset/blob/main/scripts/comparison) for more details.

![Result](https://github.com/kimihiroh/timeset/blob/main/notebooks/figures/result_formulation_comparison_base.png)

## Benchmarking
Check [`scripts/comparison`](https://github.com/kimihiroh/timeset/blob/main/scripts/benchmark) for more details.

| Tuning Method | Model (\#parameter) | TemporalNLI <br>(Accuracy) | MATRES <br>(micro-F1) | TDDiscourse <br>(micro-F1) | TORQUE <br>(Exact Match) |
|---------------|---------------------|:-----------------------:|:-----------------:|:----------------------:|:--------------------:|
| FT            | DeBERTaV3 (440M)    | 0.531                   | 0.736             | 0.439                  | 0.493                |
|               | Flan-T5 (770M)      | 0.524                   | 0.744             | 0.234                  | 0.407                |
| PEFT          | DeBERTaV3 (440M)    | 0.211                   | 0.743             | 0.403                  | 0.510                |
|               | Flan-T5 (770M)      | 0.550                   | 0.763             | 0.243                  | 0.463                |
|               | Flan-T5 (3B)        | 0.550                   | 0.750             | 0.437                  | 0.509                |
|               | Llama-2 (7B)        | 0.539                   | 0.717             | ---                     | 0.436                |
| ICL           | Llama 2 (7B)        | 0.269                   | 0.139             | 0.147                  | 0.118                |
|               | Llama 2 (13B)       | 0.336                   | 0.457             | 0.204                  | 0.086                |
|               | Llama 2 (70B)       | 0.329                   | 0.290             | 0.033                  | 0.158                |
|               | Llama 2 Chat (7B)   | 0.340                   | 0.473             | 0.214                  | 0.036                |
|               | Flan-T5 (3B)        | 0.337                   | 0.311             | 0.063                  | 0.028                |
|               | Flan-T5 (11B)       | 0.375                   | 0.386             | 0.124                  | 0.034                |
|               | T5 (3B)             | 0.0                     | 0.0               | 0.0                    | 0.0                  |
|Existing Works | --- |0.625 <br>([Mathur et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.73/)) | 0.840 <br>([Zhou et al., COLING 2022](https://aclanthology.org/2022.coling-1.174/)) | 0.511 <br>([Man et al., AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/21354)) | 0.522 <br>([Huang et al., NAACL 2022](https://aclanthology.org/2022.naacl-main.28/)) |

## Citation
If you find this work helpful in your research, please consider citing our work,
```bib
@article{hasegawa-etal-2024-formulation,
      title={Formulation Comparison for Timeline Construction using LLMs},
      author={Hasegawa, Kimihiro and Kandukuri, Nikhil and Holm, Susan and Yamakawa, Yukari and Mitamura, Teruko},
      publisher = {arXiv},
      year={2024},
      url={https://arxiv.org/abs/2403.00990},
}
```

## Issues
For any issues, questions or requests, please create a [Github Issue](https://github.com/kimihiroh/timeset/issues).
