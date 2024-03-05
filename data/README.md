# TimeSET

## Annotation preparation
Wikinews article download & preprocess: `../notebooks/preprocess-wikinews.ipynb`

Preprocessed articles in the brat format is under `./wikinews/`.

## Annotated data
Original data is under `./brat/`.

### Conversion
```bash
bash scripts/preprocess/timeset.sh
```
`json` format data is under `./preprocess/timeset/`.

To obtain only the data used in the paper: `../notebooks/sample-date.ipynb`

Sampled data is under `./preprocess/timeset-sample/`.
