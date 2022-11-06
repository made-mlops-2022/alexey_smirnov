import os
import pickle
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from src.params import TrainingParams
from src.models.model import serialize_model, predict_model, calculate_metrics, train_model
from test_feature_data import TestFeaturesModule
from generate_fake_data import generate_fake_data


class TestModelModule(TestFeaturesModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        generate_fake_data()
        tp = TrainingParams()
        self.model, self.best_params, self.best_score = \
            train_model(self.X_train, self.y_train, tp)

        self.y_pred = predict_model(self.model, self.X_test)

    def test_train_model(self):
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            self.fail('Model did not fit')
        self.assertIsInstance(self.best_params, dict)
        self.assertGreaterEqual(self.best_score['val_recall'], 0)

    def test_serialize_model(self):
        pth = 'model.pkl'
        serialize_model(self.model, pth)
        self.assertTrue(os.path.exists(pth))
        with open(pth, 'rb') as f:
            model = pickle.load(f)
        self.assertIsInstance(model, LogisticRegression)
        os.remove(pth)

    def test_predict_model(self):
        self.assertIsInstance(self.y_pred, np.ndarray)
        self.assertEqual(self.y_pred.shape, (200, ))
        self.assertEqual(list(np.unique(self.y_pred)), [0, 1])

    def test_calculate_metrics(self):
        metrics = calculate_metrics(self.y_pred, self.y_test)
        self.assertIsInstance(metrics, dict)
        self.assertGreaterEqual(metrics['recall'], 0)


if __name__ == '__main__':
    unittest.main()
