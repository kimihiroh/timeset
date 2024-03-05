# TimeSET

## Annotation preparation
```
# Wikinews article download & preprocess
../notebooks/preprocess-wikinews.ipynb
```
Preprocessed articles (unannotated) in the brat format is under `./wikinews/`.

## Annotated data
Original anntoated data in the brat format is under `./brat/`.

### Conversion
```bash
bash scripts/preprocess/timeset.sh
```
Annotated data in the `json` format is under `./preprocess/timeset/`.
```
# To obtain only the data used in the paper
../notebooks/sample-date.ipynb
```
Only the data used in the paper is under `./preprocess/timeset-sample/`.
