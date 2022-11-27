mlops_classification
==============================

# MADE MLOPS course homework #1

## _Get started_

Activate your virtual environment
```
pip3 install -r requirements.txt
```
Get the data:
```
dvc remote modify myremote gdrive_use_service_account false
dvc pull
```
Run MLFlow server
```
mlflow server --backend-store-uri sqlite:///:memory --default-artifact-root ./mlruns
```

## _Run Tests_
Generate fake data:
```
python3 tests/generate_fake_data.py
```
You can find the data and it's statistics in `tests/fake_data`

Run tests:
```
python -m unittest discover tests
```

## _Run Traning_
From `ml_prject` run:
```
python3 src/train.py hydra.job.chdir=False
```
## _Run Prediction_
From `ml_prject` run:
```
python3 src/predict.py --path_to_model <path_to_model> --path_to_transformer <path_to_transformer> --path_to_csv <path_to_csv> --path_to_prediction <path_to_prediction>
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
