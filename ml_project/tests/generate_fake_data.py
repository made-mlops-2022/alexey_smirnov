import os
import pandas as pd
from sdv.tabular import GaussianCopula


def generate_fake_data():
    fake_path = 'tests/fake_data/fake_data.csv'
    if not os.path.exists(fake_path):
        data = pd.read_csv('data/raw/heart_cleveland_upload.csv')
        model = GaussianCopula()
        model.fit(data)
        fake_data = model.sample(1000)
        fake_data.to_csv(fake_path, index=False)
