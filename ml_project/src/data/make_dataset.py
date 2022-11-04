from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from params.params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def extract_target(data: pd.DataFrame,
                   target: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = data.drop(target, axis=1)
    y = data[target].to_numpy()
    return X, y


def split_train_test_data(X: np.ndarray,
                          y: np.ndarray,
                          params: SplittingParams) -> Tuple[np.ndarray, ...]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=params.train_size, random_state=params.random_state
    )
    return X_train, X_test, y_train, y_test
