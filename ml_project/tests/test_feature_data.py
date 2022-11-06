import os
import sys
import unittest

import numpy as np

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'src'
)
sys.path.append(SOURCE_PATH)

from src.params import FeatureParams
from src.features.build_features import create_transformer, preprocess_features
from test_make_dataset import TestDataModule


class TestFeaturesModule(TestDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical = ['sex', 'cp', 'fbs', 'restecg', 'slope', 'exang', 'ca', 'thal']
        self.numerical = ['age', 'trestbps', 'chol', 'thalach', 'slope']

        feature_prms = FeatureParams(categorical=self.categorical,
                                     numerical=self.numerical,
                                     target=self.target_column)
        self.transformer = create_transformer(feature_prms)

        self.transformer.fit(self.X_train)
        self.X_train = preprocess_features(self.transformer,
                                           self.X_train)
        self.X_test = preprocess_features(self.transformer,
                                          self.X_test)

    def test_create_transformer(self):
        self.assertEqual(len(self.transformer.transformers), 2)

    def test_preprocess_features(self):
        self.assertEqual(self.X_train.shape,
                         (800, 28))
        self.assertEqual(self.X_test.shape,
                         (200, 28))

        self.assertIsInstance(self.X_train, np.ndarray)
        self.assertIsInstance(self.X_test, np.ndarray)

    def test_read_data(self):
        pass

    def test_extract_target(self):
        pass

    def test_split_data(self):
        pass


if __name__ == '__main__':
    unittest.main()
