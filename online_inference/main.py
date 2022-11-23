import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi_health import health

from config import path_to_model, path_to_transformer

app = FastAPI()

model = None
transformer = None


@app.on_event('startup')
def load_model():
    with open(path_to_transformer, 'rb') as f:
        global transformer
        transformer = pickle.load(f)

    with open(path_to_model, 'rb') as f:
        global model
        model = pickle.load(f)


@app.post('/predict')
async def predict(data):
    data_df = pd.DataFrame([data.dict()])
    X = transformer.transform(data_df)
    y = model.predict(X)
    condition = 'healthy' if not y[0] else 'sick'
    return {'condition': condition}


def check_ready():
    return model is not None and transformer is not None


async def success_handler(**kwargs):
    return 'Model is ready'


async def failure_handler(**kwargs):
    return 'Model is not ready'

app.add_api_route('/health', health([check_ready],
                  success_handler=success_handler,
                  failure_handler=failure_handler))
