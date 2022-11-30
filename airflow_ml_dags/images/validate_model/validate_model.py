import pickle
import json
import logging.config
import pandas as pd
import click
from logger import log_conf
from utils import predict_model, evaluate_model

logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


@click.command()
@click.option('--data-dir')
@click.option('--transformer-dir')
@click.option('--model-dir')
@click.option('--metric-dir')
def validate_model(data_dir: str,
                   transformer_dir: str,
                   model_dir: str,
                   metric_dir: str):
    logging.debug('Start validating test raw data! Input args: '
                  '--data-dir: %s, '
                  '--transformer-dir: %s, '
                  '--model-dir: %s, '
                  '--metric-dir: %s',
                  data_dir, transformer_dir, model_dir, metric_dir)

    try:
        test_data_raw = pd.read_csv(f'{data_dir}/data.csv')
        test_target = pd.read_csv(f'{data_dir}/target.csv')
        test_target = test_target.values[:, 0]
    except Exception as error:
        logger.error('Error loading data: %s.', data_dir)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        with open(f'{transformer_dir}/transformer.pkl', 'rb') as file:
            transformer = pickle.load(file)
    except Exception as error:
        logger.error('Error loading transformer: %s', transformer_dir)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        with open(f'{model_dir}/model.pkl', 'rb') as file:
            model_with_config = pickle.load(file)
            model = model_with_config['model']
    except Exception as error:
        logger.error('Error loading model: %s', model_dir)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        test_data_processed = transformer.transform(test_data_raw)
        test_data_processed = pd.DataFrame(test_data_processed,
                                           columns=transformer.get_feature_names_out())
    except Exception as error:
        logger.error('Error transforming. Test raw columns: %s',
                     test_data_raw.columns)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        predictions = predict_model(model, test_data_processed)
    except Exception as error:
        logger.error('Error while predicting values. '
                     'Test processed columns: %s',
                     test_data_processed.columns)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        test_scores = evaluate_model(predictions, test_target)
    except Exception as error:
        logger.error('Error while evaluating test data. '
                     'True shape: %s, predicted shape: %s',
                     test_target.shape, predictions.shape)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        with open(f'{metric_dir}/test_score.json', 'w') as file:
            json.dump({'test': test_scores}, file)
    except Exception as error:
        logger.error('Error while saving test score. Path: %s',
                     metric_dir)
        logger.error('Error message: %s', error)
        raise Exception


if __name__ == '__main__':
    validate_model()
