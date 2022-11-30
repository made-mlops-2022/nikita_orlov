import os
import logging.config
import numpy as np
import pandas as pd
import click
from logger import log_conf


PATH = 'data.csv'
DATASET_LEN = 200
SEP = ','
logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')


@click.command("generate_data")
@click.option("--output-dir")
def generate_data(output_dir: str):
    logging.debug('Start generating data! Input args: '
                  '--output-dir: %s', output_dir
                  )
    df_init = pd.read_csv(PATH)
    df_shuffle = df_init.iloc[np.random.permutation(len(df_init) - 1)]
    df_shuffle.reset_index(drop=True)
    df_shuffle = df_shuffle.iloc[:DATASET_LEN]
    x_train = df_shuffle.iloc[:, :-1]
    y_train = df_shuffle.iloc[:, -1]

    logger.debug('x_train shape: %s', x_train.shape)
    logger.debug('y_train shape: %s', y_train.shape)

    os.makedirs(output_dir, exist_ok=True)
    try:
        x_train.to_csv(f'{output_dir}/data.csv', index=None)
        y_train.to_csv(f'{output_dir}/target.csv', index=None)
    except Exception as error:
        logger.error('Error saving file in output-dir %s', output_dir)
        logger.error('Error message: %s', error)
        raise Exception


if __name__ == '__main__':
    generate_data()
