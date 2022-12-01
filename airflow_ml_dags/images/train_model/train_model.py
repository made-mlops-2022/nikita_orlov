import logging.config
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import log_conf
from utils import train_model,\
    evaluate_model,\
    predict_model,\
    serialize,\
    load_config

logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


@click.command()
@click.option('--data-dir')
@click.option('--config-path')
@click.option('--model-dir')
@click.option('--metric-dir')
def train(data_dir: str, config_path: str, model_dir: str, metric_dir: str):
    logger.debug('Start training data! Input args: '
                  '--data-dir: %s, '
                  '--config-path: %s, '
                  '--model-dir: %s, '
                  '--metric-dir: %s',
                  data_dir, config_path, model_dir, metric_dir)
    try:
        config = load_config(config_path)
        splitting_config = config.splitting_config
        model_config = config.model_config
        logger.debug('Config loaded: %s', config)
    except Exception as error:
        logger.error('Error loading config: %s', config_path)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        data = pd.read_csv(f'{data_dir}/data.csv')
        target = pd.read_csv(f'{data_dir}/target.csv')
        logger.debug('Data shape: %s, target shape: %s',
                     data.shape, target.shape)
    except Exception as error:
        logger.error('Error loading data: %s', data_dir)
        logger.error('Error message: %s', error)
        raise Exception

    data_pack = train_test_split(data,
                                 target,
                                 test_size=splitting_config.test_size,
                                 random_state=splitting_config.random_state,
                                 shuffle=splitting_config.shuffle)
    train_data, val_data, train_target, val_target = data_pack
    train_target = train_target.values[:, 0]
    val_target = val_target.values[:, 0]

    assert len(train_data) == len(train_target)
    assert len(val_data) == len(val_target)
    assert len(train_data) + len(val_data) == len(data)
    assert train_data.shape[1] == val_data.shape[1] == data.shape[1]

    logger.debug('Train data shape: %s', train_data.shape)
    logger.debug('Train target shape: %s', train_target.shape)
    logger.debug('Validation data shape: %s', val_data.shape)
    logger.debug('Validation target shape: %s', val_target.shape)

    try:
        model = train_model(train_data, train_target, model_config)
    except NotImplementedError as error:
        logger.error('Wrong model name: %s not implemented',
                     model_config.model)
        logger.error('Error message: %s', error)
        raise NotImplementedError

    predictions_train = predict_model(model, train_data)
    score_train = evaluate_model(predictions_train, train_target)
    logger.info('Train score: %s', score_train)

    predictions_val = predict_model(model, val_data)
    score_val = evaluate_model(predictions_val, val_target)
    logger.info('Validation score: %s', score_val)

    try:
        serialize({'train': score_train},
                  f'{metric_dir}/train_score.json',
                  'json')

        serialize({'val': score_val},
                  f'{metric_dir}/val_score.json',
                  'json')
    except Exception as error:
        logger.error('Cant save file: %s', metric_dir)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        serialize({'model': model, 'train_config': model_config},
                  f'{model_dir}/model.pkl',
                  'pickle')
    except Exception as error:
        logger.error('Cant save file: %s', model_dir)
        logger.error('Error message: %s', error)
        raise Exception


if __name__ == '__main__':
    train()
