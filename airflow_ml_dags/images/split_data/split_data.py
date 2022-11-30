import os
from dataclasses import dataclass
import logging.config
import yaml
import pandas as pd
import click
from sklearn.model_selection import train_test_split
from logger import log_conf

logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


@dataclass()
class SplittingParams:
    test_size: float
    random_state: int
    shuffle: bool


def load_params(path: str) -> SplittingParams:
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    config = SplittingParams(**data)
    return config


@click.command()
@click.option('--input-dir')
@click.option('--config-path')
@click.option('--train-dir')
@click.option('--test-dir')
def split_data(input_dir: str, config_path: str, train_dir: str, test_dir: str):
    logger.debug('Start splitting! --input-dir: %s, '
                 '--config-path: %s, '
                 '--train-dir: %s ,'
                 '--test-dir: %s',
                 input_dir, config_path, train_dir, test_dir)
    try:
        data = pd.read_csv(f'{input_dir}/data.csv')
        target = pd.read_csv(f'{input_dir}/target.csv')
        logger.debug('Data init shape: %s, target shape: %s',
                     data.shape, target.shape)
    except Exception as error:
        logger.error('Cant load from %s', input_dir)
        logger.error('Error message: %s', error)
        raise Exception

    config = load_params(config_path)
    logger.debug('Config loaded: %s', config)
    data_pack = train_test_split(data,
                                 target,
                                 test_size=config.test_size,
                                 random_state=config.random_state,
                                 shuffle=config.shuffle)

    train_data, test_data, train_target, test_target = data_pack
    assert len(train_data) == len(train_target)
    assert len(test_data) == len(test_target)
    assert len(train_data) + len(test_data) == len(data)
    assert train_data.shape[1] == test_data.shape[1] == data.shape[1]
    assert train_target.shape[1] == test_target.shape[1] == target.shape[1]
    logger.debug('Train data shape: %s', train_data.shape)
    logger.debug('Train target shape: %s', train_target.shape)
    logger.debug('Test data shape: %s', test_data.shape)
    logger.debug('Test target shape: %s', test_target.shape)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    try:
        train_data.to_csv(f'{train_dir}/data.csv', index=None)
        train_target.to_csv(f'{train_dir}/target.csv', index=None)
    except Exception as error:
        logger.error('Cant save file: %s', train_dir)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        test_data.to_csv(f'{test_dir}/data.csv', index=None)
        test_target.to_csv(f'{test_dir}/target.csv', index=None)
    except Exception as error:
        logger.error('Cant save file: %s', train_dir)
        logger.error('Error message: %s', error)
        raise Exception


if __name__ == '__main__':
    split_data()
