import json

import hydra
import mlflow
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from data.make_dataset import extract_target, read_data, split_data
from params.params import TrainConfig
from features.build_features import create_transformer, process_features
from logger.logger import logger
from models.model import (predict_model,
                          train_model,
                          save_model,
                          calculate_metrics)


cs = ConfigStore.instance()
cs.store(name='train', node=TrainConfig)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def run_train_pipeline(params: TrainConfig):

    # preprocessing pipeline
    params = instantiate(params, _convert_='partial')
    logger.info(f'Starting train pipeline with params {params}')
    data = read_data(params.path_to_input_data)
    logger.info(f'Successfully read DataFrame, shape is {data.shape}')

    X, y = extract_target(data, params.feature_params.target_column)

    X_train, X_test, y_train, y_test = split_data(X,
                                                  y,
                                                  params.splitting_params)
    logger.info('Split data to train and test...')
    X_test.to_csv(params.path_to_test_data)
    logger.info(f'Saved unlabeled test data to {params.path_to_test_data}')

    logger.info('Preprocessing features... ')
    transformer = create_transformer(params.feature_params)
    transformer.fit(X_train)
    X_train = process_features(transformer, X_train)
    df_train_processed = np.concatenate([X_train,
                                         y_train[..., np.newaxis]],
                                        axis=-1)
    pd.DataFrame(df_train_processed).to_csv(params.model.processed_data_path,
                                            index=False)
    logger.info(f'Saved features to {params.model.processed_data_path}')

    save_model(transformer,
               params.model.transformer_model_path)
    logger.info(f'Saved transform to {params.model.path_to_transformer}')

    # mlflow traning pipeline
    mlflow.set_tracking_uri(params.mlflow_url)
    with mlflow.start_run(run_name=params.mlflow_run_name):
        logger.info('Start training model...')
        if params.model.train_params.grid_search:
            model, bst_params, metrics = train_model(X_train,
                                                     y_train,
                                                     params.model.train_params)
            logger.info(f'Best params are {bst_params}')
            for hp in bst_params:
                mlflow.log_param(hp, bst_params[hp])
            logger.info(f'Validation matrics: {metrics}')
            for metric in metrics:
                mlflow.log_metric(metric, metrics[metric])
        else:
            model = train_model(X_train, y_train, params.model.train_params)

        # calculating metrics
        logger.info('Calculating metrics...')
        y_pred = predict_model(model,
                               process_features(transformer, X_test))
        metrics = calculate_metrics(y_pred,
                                    y_test)
        for metric in metrics:
            mlflow.log_metric(metric,
                              metrics[metric])
        logger.info(f'Metrics on test data {metrics}')
        with open(params.model.metric_json_path, 'w') as metric_file:
            if params.model.train_params.grid_search:
                json.dump({**metrics, **metrics}, metric_file)
            else:
                json.dump(metrics, metric_file)

        # saving model
        logger.info(f'Saving model to {params.model.model_path} ...')
        save_model(model,
                   params.model.model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classification_model")
        logger.info('Successfully save model')


if __name__ == '__main__':
    run_train_pipeline()
