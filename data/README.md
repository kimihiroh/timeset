# TimeSET

## Annotation preparation
```
# Wikinews article download & preprocess
../notebooks/preprocess-wikinews.ipynb
```
Preprocessed articles (unannotated) in the brat format is under `./wikinews/`.

## Annotated data
Original anntoated data in the brat format is under `./brat/`.

## Conversion
```bash
bash scripts/preprocess/timeset.sh
```
Annotated data in the `json` format is under `./preprocess/timeset/`.
```
# To obtain only the data used in the paper
../notebooks/sample-date.ipynb
```
Only the data used in the paper is under `./preprocess/timeset-sample/`.

# Benchmark
## Download data
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
## Preprocess
```bash
# TORQUE
bash scripts/preprocess/torque.sh
# MATRES
bash scripts/preprocess/matres.sh
# temporal nli (TempEval3)
# follow their instruction, and # TODO: add customized preprocessing script & instruction
bash scripts/preprocess/temporal_nli.sh
# TDDiscourse (TDDMan)
bash scripts/preprocess/tddiscourse.sh
```
