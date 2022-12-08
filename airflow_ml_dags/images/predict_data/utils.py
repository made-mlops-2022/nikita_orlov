from typing import Union, Dict
from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

SklearnModel = Union[LogisticRegression, RandomForestClassifier]


@dataclass()
class ModelConfig:
    model: str
    random_state: int
    model_params: Union[Dict, None]


def predict_model(model: SklearnModel, data: pd.DataFrame):
    predictions = model.predict(data)
    return predictions


def evaluate_model(predictions: np.ndarray, target: np.ndarray)\
        -> Dict[str, float]:
    if predictions.shape != target.shape:
        raise RuntimeError
    scores = {'accuracy': round(accuracy_score(target, predictions), 4),
              'f1': round(f1_score(target, predictions), 4),
              'roc_auc': round(roc_auc_score(target, predictions), 4)}
    return scores
