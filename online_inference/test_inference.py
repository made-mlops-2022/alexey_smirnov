import json
import logging

from random import randint

import requests
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

def main():
    for _ in range(10):
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
        response = requests.post(
            'http://127.0.0.1:8000/predict',
            json.dumps(request)
        )

        logger.info(f'Request: {request}')
        logger.info('Response:')
        logger.info(f'Status Code: {response.status_code}')
        logger.info(f'Message: {response.json()}')
        logger.info('-------------------------------------')
        time.sleep(1)


if __name__ == '__main__':
    main()
