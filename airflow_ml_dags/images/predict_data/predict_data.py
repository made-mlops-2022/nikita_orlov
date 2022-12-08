import pickle
import os
import logging.config
import click
import pandas as pd
from logger import log_conf
from utils import predict_model, evaluate_model


logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


@click.command()
@click.option('--data-dir')
@click.option('--prediction-dir')
@click.option('--transformer-dir')
@click.option('--model-dir')
def predict(data_dir: str, prediction_dir: str, transformer_dir: str,  model_dir: str):
    logger.debug('Start predicting raw data! Input args: '
                  '--data-dir: %s, '
                  '--prediction-dir: %s, '
                  '--transformer-dir: %s, '
                  '--model-dir: %s, ',
                  data_dir, prediction_dir, transformer_dir, model_dir)

    try:
        test_data_raw = pd.read_csv(f'{data_dir}/data.csv')
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
    #
    # y_true = pd.read_csv(f'{data_dir}/target.csv').values[:, 0]
    # score = evaluate_model(predictions, y_true)
    # print(score)

    os.makedirs(prediction_dir, exist_ok=True)
    with open(f'{prediction_dir}/predictions.csv', 'w') as file:
        file.write('\n'.join(map(str, predictions)))


if __name__ == '__main__':
    # data_dir = '/home/lolvista/MADE/mlops_course/mlops_made_for_org/airflow_ml_dags/data/raw/2022-11-21'
    # prediction_dir = '/home/lolvista/MADE/mlops_course/mlops_made_for_org/airflow_ml_dags/data/predictions/2022-11-21'
    # transformer_dir = '/home/lolvista/MADE/mlops_course/mlops_made_for_org/airflow_ml_dags/data/transformers/2022-11-21'
    # model_dir = '/home/lolvista/MADE/mlops_course/mlops_made_for_org/airflow_ml_dags/data/models/2022-11-21'
    #
    # predict(data_dir, prediction_dir, transformer_dir, model_dir)
    predict()
