import unittest

import numpy as np
import pandas as pd

from src.data.make_dataset import extract_target, read_data, split_data
from src.params import SplittingParams


class TestDataModule(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_file = 'tests/fake_data/fake_data.csv'
        self.target_column = 'condition'

        self.data = read_data(self.input_file)
        self.X, self.y = extract_target(self.data, self.target_column)

        self.split_params = SplittingParams()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            split_data(self.X, self.y, self.split_params)

    def test_read_data(self):
        self.assertEqual(self.data.shape, (self.data.shape[0], 14))
        self.assertIn(self.target_column, self.data.columns)

    def test_extract_target(self):
        self.assertIsInstance(self.X, pd.DataFrame)
        self.assertIsInstance(self.y, np.ndarray)
        self.assertEqual(self.X.shape, (self.data.shape[0], 13))
        self.assertEqual(self.y.shape, (self.data.shape[0], ))

    def test_split_data(self):

        self.assertIsInstance(self.X_train, pd.DataFrame)
        self.assertIsInstance(self.y_train, np.ndarray)
        self.assertIsInstance(self.X_test, pd.DataFrame)
        self.assertIsInstance(self.y_test, np.ndarray)

        self.assertEqual(self.X_train.shape, (800, 13))
        self.assertEqual(self.y_train.shape, (800, ))
        self.assertEqual(self.X_test.shape, (200, 13))
        self.assertEqual(self.y_test.shape, (200, ))


if __name__ == '__main__':
    unittest.main()
