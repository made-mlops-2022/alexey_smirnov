import json
from random import randint

import pytest
from fastapi.testclient import TestClient

from app import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_predict_endpoint1():
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


def test_predict_endpoint2():
    request = {
        'age': 67,
        'sex': 1,
        'cp': 2,
        'trestbps': 123,
        'chol': 333,
        'fbs': 1,
        'restecg': 1,
        'thalach': 112,
        'exang': 0,
        'oldpeak': 5.5,
        'slope': 2,
        'ca': 0,
        'thal': 1
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'prediction': 'sick'}


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
        # 'thalach': randint(0, 250),  <----------------------
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


def test_fbs_outrange():
    request = {
        'age': randint(0, 110),
        'sex': randint(0, 1),
        'cp': randint(0, 3),
        'trestbps': randint(0, 250),
        'chol': randint(0, 700),
        # 'fbs': randint(0, 1),
        'fbs': 2,                   # <----------------------
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


def test_age_outrange():
    request = {
        'age': 112,    # <-----------------------
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


def test_oldpeak_outrange():
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
        'oldpeak': 10.5,      # <-----------------------
        'slope': randint(0, 2),
        'ca': randint(0, 3),
        'thal': randint(0, 2)
    }
    response = client.post(
        '/predict',
        json.dumps(request)
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == 'wrong oldpeak value'
