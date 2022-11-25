import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import logging.config
from src.logger import log_conf


predict_app = FastAPI()
PIPELINE_PATH = 'models/model.pkl'
FEATURES = None
PIPELINE = None

logging.config.dictConfig(log_conf)
logger = logging.getLogger('file_stream')

class DatasetType(BaseModel):
    age: List[float]
    trestbps: List[float]
    chol: List[float]
    thalach: List[float]
    oldpeak: List[float]
    sex: List[float]
    cp: List[float]
    fbs: List[float]
    restecg: List[float]
    exang: List[float]
    slope: List[float]
    ca: List[float]
    thal: List[float]


class ResponseType(BaseModel):
    preds: List[int]


def load_pipeline(path: str):
    global PIPELINE
    global FEATURES

    try:
        with open(path, 'rb') as file:
            train_obj = pickle.load(file)
    except FileNotFoundError:
        logger.error(f'Model path doesnt exist: {PIPELINE_PATH}')
        sys.exit(1)

    PIPELINE = train_obj['pipeline']

    train_config = train_obj['train_config']
    categorical_cols = train_config.feature_params.categorical_columns
    numerical_cols = train_config.feature_params.numerical_columns
    FEATURES = categorical_cols + numerical_cols


@predict_app.on_event('startup')
async def load_model():
    load_pipeline(PIPELINE_PATH)
    logger.debug('Model has been loaded!')


def predict_core(input_data: DatasetType) -> ResponseType:
    data = pd.DataFrame.from_dict(input_data.dict())
    if FEATURES is None:
        logger.error(f'Model has not been loaded! {PIPELINE_PATH}')
        raise HTTPException(status_code=400, detail='Model has not been loaded!')
    for column in FEATURES:
        if column not in data.columns:
            logger.error(f'Feature {column} does not exist in input data! User columns: {list(data.columns)}')
            raise KeyError(f'Feature {column} does not exist in input data!')

    data = data[FEATURES]
    predictions = PIPELINE.predict(data)
    response = ResponseType(preds=list(predictions))
    return response


@predict_app.get('/')
async def root():
    return {'message': 'Hello world!'}


@predict_app.get('/health')
async def health() -> int:
    print(PIPELINE)
    print(FEATURES)
    if PIPELINE is not None and FEATURES is not None:
        return 200
    # return 404
    raise HTTPException(status_code=400, detail='Model has not been loaded!')


@predict_app.post('/predict')
async def predict(item: DatasetType) -> ResponseType:
    # try:
    preds = predict_core(item)
    # except:

    return preds
