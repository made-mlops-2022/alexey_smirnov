import json
from random import randint

import pytest
from fastapi.testclient import TestClient

from app import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_predict_endpoint():
    request = {
        'age': randint(0, 110),
        'sex': randint(0, 1),
        'cp': randint(0, 3),
        'trestbps': randint(0, 250),
        'chol': randint(0, 700),
        'fbs': randint(0, 1),
        'restecg': randint(0, 2),
        'thalach': randint(0, 250),
        'exang': randint(0, 1),
        'oldpeak': randint(0, 80) / 10,
        'slope': randint(0, 2),
        'ca': randint(0, 3),
        'thal': randint(0, 2)
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'condition': 'sick'}


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == 'Model is ready'


def test_miss_field():
    request = {
        'age': randint(0, 110),
        'sex': randint(0, 1),
        'cp': randint(0, 3),
        'trestbps': randint(0, 250),
        'chol': randint(0, 700),
        'fbs': randint(0, 1),
        'restecg': randint(0, 2),
        # 'thalach': randint(0, 250),
        'exang': randint(0, 1),
        'oldpeak': randint(0, 80) / 10,
        'slope': randint(0, 2),
        'ca': randint(0, 3),
        'thal': randint(0, 2)
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'field required'


def test_cat_field_outrange():
    request = {
        'age': randint(0, 110),
        'sex': randint(0, 1),
        'cp': randint(0, 3),
        'trestbps': randint(0, 250),
        'chol': randint(0, 700),
        # 'fbs': randint(0, 1),
        'fbs': 2,    # <---------------------
        'restecg': randint(0, 2),
        'thalach': randint(0, 250),
        'exang': randint(0, 1),
        'oldpeak': randint(0, 80) / 10,
        'slope': randint(0, 2),
        'ca': randint(0, 3),
        'thal': randint(0, 2)
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'unexpected value; permitted: 0, 1'


def test_num_field_outrange():
    request = {
        'age': 112,  # <--------------------
        'sex': randint(0, 1),
        'cp': randint(0, 3),
        'trestbps': randint(0, 250),
        'chol': randint(0, 700),
        'fbs': randint(0, 1),
        'restecg': randint(0, 2),
        'thalach': randint(0, 250),
        'exang': randint(0, 1),
        'oldpeak': randint(0, 80) / 10,
        'slope': randint(0, 2),
        'ca': randint(0, 3),
        'thal': randint(0, 2)
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'wrong age value'
