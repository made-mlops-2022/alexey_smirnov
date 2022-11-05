import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from params.params import FeatureParams


def create_categorical_pipeline() -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                     ('ohe', OneHotEncoder())])


def create_numerical_pipeline() -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='mean')),
                     ('scaler', StandardScaler())])


def process_features(transformer: ColumnTransformer,
                     data: pd.DataFrame) -> np.ndarray:
    return transformer.transform(data)


def create_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                create_categorical_pipeline(),
                params.categorical,
            ),
            (
                'numerical_pipeline',
                create_numerical_pipeline(),
                params.numerical,
            ),
        ]
    )
    return transformer
