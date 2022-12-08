import os
import pickle
import logging.config
from dataclasses import dataclass
from typing import List
import pandas as pd
import click
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from logger import log_conf


logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


@dataclass()
class FeatureParams:
    numerical_columns: List[str]
    categorical_columns: List[str]
    target_column: str
    fill_na_numerical_strategy: str
    fill_na_categorical_strategy: str


def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    config = FeatureParams(**config)
    return config


def build_numerical_pipeline(params: FeatureParams) -> Pipeline:
    fill_strategy = params.fill_na_numerical_strategy
    pipe = Pipeline([('impute',
                      SimpleImputer(strategy=fill_strategy)),
                     ('scaler',
                      StandardScaler())])
    return pipe


def build_categorical_pipeline(params: FeatureParams) -> Pipeline:
    fill_strategy = params.fill_na_categorical_strategy
    pipe = Pipeline([('impute',
                      SimpleImputer(strategy=fill_strategy)),
                     ('OHE',
                      OneHotEncoder())])
    return pipe


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    numerical_pipe = build_numerical_pipeline(params)
    categorical_pipe = build_categorical_pipeline(params)
    transformer = ColumnTransformer([('numerical_step',
                                      numerical_pipe,
                                      params.numerical_columns),
                                     ('categorical_step',
                                      categorical_pipe,
                                      params.categorical_columns)])
    return transformer


def transform_data(data: pd.DataFrame, transformer: ColumnTransformer)\
        -> pd.DataFrame:
    df_transformed = pd.DataFrame(transformer.fit_transform(data),
                                  columns=transformer.get_feature_names_out())
    return df_transformed


def transform_target(data: pd.DataFrame) -> pd.DataFrame:
    return data


@click.command()
@click.option('--data-dir')
@click.option('--config-path')
@click.option('--output-dir')
@click.option('--transformer-dir')
def prepare_data(data_dir: str,
                 config_path: str,
                 output_dir: str,
                 transformer_dir: str):
    logging.debug('Start processing data! Input args: '
                  '--data-dir: %s, '
                  '--config-path: %s, '
                  '--output-dir: %s'
                  '--transformer-dir: %s',
                  data_dir, config_path, output_dir, transformer_dir
                  )
    try:
        data = pd.read_csv(f'{data_dir}/data.csv')
        target = pd.read_csv(f'{data_dir}/target.csv')
        logger.debug('Read data and target from %s', data_dir)
        logger.debug('Data init shape: %s, target shape %s',
                     data.shape, target.shape)
    except Exception as error:
        logger.error('Cant open file: %s', data_dir)
        logger.error('Error message: %s', error)
        raise Exception

    try:
        config = load_config(config_path)
        logger.debug('Config loaded: %s', config)
    except Exception as error:
        logger.error('Cant open file: %s', config_path)
        logger.error('Error message: %s', error)
        raise Exception

    transformer = build_transformer(config)
    processed_data = transform_data(data, transformer)
    processed_target = transform_target(target)
    logger.debug('Transformed data shape: %s, transformed target shape: %s',
                 processed_data.shape, processed_target.shape)

    os.makedirs(transformer_dir, exist_ok=True)
    try:
        with open(f'{transformer_dir}/transformer.pkl', 'wb') as file:
            pickle.dump(transformer, file)
    except Exception as error:
        logger.error('Cant save transformer: %s', transformer_dir)
        logger.error('Error message: %s', error)
        raise Exception

    os.makedirs(output_dir, exist_ok=True)
    try:
        processed_data.to_csv(f'{output_dir}/data.csv', index=None)
        processed_target.to_csv(f'{output_dir}/target.csv', index=None)
    except Exception as error:
        logger.error('Cant save file: %s', output_dir)
        logger.error('Error message: %s', error)
        raise Exception


if __name__ == '__main__':
    prepare_data()
