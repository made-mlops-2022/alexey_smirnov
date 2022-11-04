from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

from params.params import TrainingParams


def train_model(X_train: np.ndarray, 
                y_train: np.ndarray,
                train_params: TrainingParams
                ) -> Tuple[LogisticRegression, ...]:
    
    model = LogisticRegression(random_state=train_params.random_state, solver='liblinear')
    param_grid = {'C': np.logspace(-5, 5, 10),
                  'penalty': ['l1', 'l2']}

    if train_params.grid_search:
        model_gscv = GridSearchCV(model, param_grid, scoring='recall', cv=5)
        model_gscv.fit(X_train, y_train)

        model = model_gscv.best_estimator_
        return model_gscv.best_estimator_, model_gscv.best_params_,  \
            {'val_recall': model_gscv.best_score_}
    else:
        model.fit(X_train, y_train)
        return model


def predict_model(model: LogisticRegression, 
                  X: np.ndarray) -> np.ndarray:
    return model.predict(X)
