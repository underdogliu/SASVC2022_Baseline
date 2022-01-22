## Introduction
This repository contains several materials that supplements the Spoofing-Aware Speaker Verification (SASV) Challenge 2022 including:
- calculating metrics;
- extracting speaker/spoofing embeddings from pre-trained models;
- training/evaluating Baseline2 in the evaluation plan. 

### Data preparation
The ASVspoof2019 LA dataset [1] can be downloaded using the scipt in AASIST [2] repository
```
python ./aasist/download_dataset.py
```

### Speaker & spoofing embedding extraction

```
mkdir -p data/_meta/
cp ./LA/*protocols/*.txt ./data/_meta/

python save_embeddings.py
```

### Baseline 2 Training
```
python main.py
```
