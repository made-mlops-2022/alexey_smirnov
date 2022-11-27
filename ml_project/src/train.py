import json

import hydra
import mlflow
import numpy as np
import pandas as pd
from data.make_dataset import extract_target, read_data, split_data
from features.build_features import create_transformer, preprocess_features
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from logger.logger import logger
from models.model import (calculate_metrics, predict_model, serialize_model,
                          train_model)
from params.params import TrainConfig

cs = ConfigStore.instance()
cs.store(name='train', node=TrainConfig)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def run_train_pipeline(params: TrainConfig):

    # preprocessing pipeline
    params = instantiate(params, _convert_='partial')
    logger.info(f'Starting train pipeline with params {params}')
    data = read_data(params.input_data_path)
    logger.info(f'Successfully read DataFrame, shape is {data.shape}')

    X, y = extract_target(data, params.feature_params.target)

    logger.info('Spliting data to train and test...')
    X_train, X_test, y_train, y_test = split_data(X,
                                                  y,
                                                  params.splitting_params)
    logger.info(f'X_train.shape is {X_train.shape}, \
                  y_train.shape is {y_train.shape}')
    logger.info(f'X_test.shape is {X_train.shape}, \
                  y_test.shape is {y_train.shape}')
    X_test.to_csv(params.test_data_path)
    logger.info(f'Saved test data to {params.test_data_path}')

    logger.info('Preprocessing features... ')
    transformer = create_transformer(params.feature_params)
    transformer.fit(X_train)
    X_train = preprocess_features(transformer,
                                  X_train)
    df_train_processed = np.concatenate([X_train,
                                         y_train[..., np.newaxis]],
                                        axis=-1)
    pd.DataFrame(df_train_processed).to_csv(params.model.processed_data_path,
                                            index=False)
    logger.info(f'Saved features to {params.model.processed_data_path}')

    serialize_model(transformer,
                    params.model.transformer_model_path)
    logger.info(f'Saved transform to {params.model.transformer_model_path}')

    # mlflow traning pipeline
    mlflow.set_tracking_uri(params.mlflow_url)
    with mlflow.start_run(run_name=params.mlflow_run_name):
        logger.info('Start training model...')
        if params.model.train_params.grid_search:
            logger.info('Using grid search for best model hyperparameters...')
            model, bst_params, val_metrics = train_model(
                X_train,
                y_train,
                params.model.train_params
            )
            logger.info(f'Best params are {bst_params}')

            for prm in bst_params:
                mlflow.log_param(prm, bst_params[prm])
            logger.info(f'Validation matrics: {val_metrics}')

            for metric in val_metrics:
                mlflow.log_metric(metric,
                                  val_metrics[metric])
        else:
            model = train_model(X_train,
                                y_train,
                                params.model.train_params)

        # calculating metrics
        logger.info('Calculating metrics...')
        y_pred = predict_model(model,
                               preprocess_features(transformer,
                                                   X_test))
        metrics = calculate_metrics(y_pred,
                                    y_test)
        for metric in metrics:
            mlflow.log_metric(metric,
                              metrics[metric])
        logger.info(f'Metrics on test data: {metrics}')

        with open(params.model.metric_json_path, 'w') as metric_file:
            if params.model.train_params.grid_search:
                json.dump({**val_metrics, **metrics}, metric_file)
            else:
                json.dump(metrics, metric_file)

        # saving model
        logger.info(f'Saving model to {params.model.model_path} ...')
        serialize_model(model,
                        params.model.model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classification_model"
        )
        logger.info('Successfully save model')


if __name__ == '__main__':
    run_train_pipeline()
