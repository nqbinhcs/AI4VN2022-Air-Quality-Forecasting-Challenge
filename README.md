### The directory structure

```
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   ├── raw            <- The original, immutable data dump.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── external       <- Data from third party sources.
│
├── saved             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is task name SHKPA-XX (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `SHKPA-67-mms-test-LSTM-model-on-all-electrolyses`.
│
├── docs               <- Questions and some other related documentation
│
├── results            <- submission.
├── config            <- config file.
│
├── .gitignore         <- Avoids uploading data, credentials, outputs, system files etc
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
|
└── src                <- Source code for use in this project.
```
### Command
Training
```
python src/train.py --config config/config.yaml
```

Inference result
```
python src/inference.py --config config/config.yaml
```
