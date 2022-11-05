import pickle

import click
import pandas as pd

from data import read_data
from features.build_features import process_features
from logger.logger import logger
from models.model import predict_model


@click.command()
@click.option('--model_path', type=click.Path(exists=True),
              default='models/output_model.pkl')
@click.option('--transformer_path', type=click.Path(exists=True),
              default='models/transformers/transformer.pkl')
@click.option('--test_data_path', type=click.Path(exists=True),
              default='data/test/heart_cleveland_upload_test.csv')
@click.option('--predictions_path', type=click.Path(exists=False),
              default='models/predictions/predictions.csv')
def run_predict_pipeline(model_path: str,
                         transformer_path: str,
                         test_data_path: str,
                         predictions_path: str):

    logger.info('Starting predict pipeline with params ...')
    data = read_data(test_data_path)
    logger.info(f'Successfully read test data, shape is {data.shape}')

    logger.info('Transforming features... ')
    with open(transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    X = process_features(transformer, data)
    logger.info('Successfully transformed features')

    logger.info('Making predictions... ')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y = predict_model(model, X)
    logger.info(f'Make prediction using {type(model).__name__}')

    pd.DataFrame(y).to_csv(predictions_path, index=False)
    logger.info(f'Saved predictions to {predictions_path}')


if __name__ == '__main__':
    run_predict_pipeline()
