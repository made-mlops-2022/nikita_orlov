import os
import pickle
import json
from typing import Union, Dict
from dataclasses import dataclass
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


SklearnModel = Union[LogisticRegression, RandomForestClassifier]


@dataclass()
class ModelConfig:
    model: str
    random_state: int
    model_params: Union[Dict, None]


@dataclass()
class SplittingConfig:
    test_size: float
    random_state: int
    shuffle: bool


@dataclass()
class TrainingConfig:
    model_config: ModelConfig
    splitting_config: SplittingConfig


def load_config(path: str) -> TrainingConfig:
    with open(path, 'rb') as file:
        data = yaml.safe_load(file)
    model_config = ModelConfig(**data['model_config'])
    splitting_config = SplittingConfig(**data['splitting_config'])
    config = TrainingConfig(model_config=model_config,
                            splitting_config=splitting_config)
    return config


def train_model(data: pd.DataFrame, target: np.ndarray, config: ModelConfig)\
        -> SklearnModel:
    if config.model_params is None:
        config.model_params = {}
    if config.model == 'RandomForestClassifier':
        model = RandomForestClassifier(**config.model_params)
    elif config.model == 'LogisticRegression':
        model = LogisticRegression(**config.model_params)
    else:
        raise NotImplementedError

    model.fit(data, target)
    return model


def predict_model(model: SklearnModel, data: pd.DataFrame):
    predictions = model.predict(data)
    return predictions


def serialize(obj: object, path: str, data_type: str = 'pickle'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if data_type == 'pickle':
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    elif data_type == 'json':
        with open(path, 'w') as file:
            json.dump(obj, file)
    else:
        raise NotImplementedError


def evaluate_model(predictions: np.ndarray, target: np.ndarray)\
        -> Dict[str, float]:
    if predictions.shape != target.shape:
        raise RuntimeError
    scores = {'accuracy': round(accuracy_score(target, predictions), 4),
              'f1': round(f1_score(target, predictions), 4),
              'roc_auc': round(roc_auc_score(target, predictions), 4)}
    return scores
