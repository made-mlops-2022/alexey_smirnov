import os
import pickle
import sys
import unittest

from hydra import compose, initialize
from hydra.utils import instantiate
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from src.train import run_train_pipeline
from src.predict import run_predict_pipeline

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'src'
)
sys.path.append(SOURCE_PATH)


class TestTrainPredict(unittest.TestCase):
    def test_train_predict(self):

        with initialize(version_base=None, config_path='../configs'):
            params = compose(config_name="config")

        run_train_pipeline(params)

        self.assertTrue(os.path.exists(params.test_data_path))
        self.assertTrue(os.path.exists(params.model.model_path))
        self.assertTrue(os.path.exists(params.model.metric_json_path))
        self.assertTrue(os.path.exists(params.model.processed_data_path))
        self.assertTrue(os.path.exists(params.model.transformer_model_path))

        with open(params.model.model_path, 'rb') as file:
            model = pickle.load(file)
        self.assertIsInstance(model, LogisticRegression)

        with open(params.model.transformer_model_path, 'rb') as f:
            model = pickle.load(f)
        self.assertIsInstance(model, ColumnTransformer)

        run_predict_pipeline.callback(model_path=params.model.model_path,
                                      transformer_path=params.model.transformer_model_path,
                                      test_data_path=params.test_data_path,
                                      predictions_path='models/prediction.csv')
        self.assertTrue(os.path.exists('models/prediction.csv'))


if __name__ == '__main__':
    unittest.main()
