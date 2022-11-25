from fastapi.testclient import TestClient
from app import predict_app
from unittest import TestCase
import unittest
from copy import deepcopy

DATA = {'age': [69, 69, 66, 65, 64],
        'sex': [1, 0, 0, 1, 1],
        'cp': [0, 0, 0, 0, 0],
        'trestbps': [1600, 1400, 1500, 1380, 1100],
        'chol': [234, 239, 226, 282, 211],
        'fbs': [1, 0, 0, 1, 0],
        'restecg': [2, 0, 0, 2, 2],
        'thalach': [131, 151, 114, 174, 144],
        'exang': [0, 0, 0, 0, 1],
        'oldpeak': [0.1, 1.8, 2.6, 1.4, 1.8],
        'slope': [1, 0, 2, 1, 1],
        'ca': [1, 2, 0, 1, 0],
        'thal': [0, 0, 0, 0, 0],
        'condition': [0, 0, 0, 1, 0]}


class TestHealth(TestCase):
    def test_correct(self):
        with TestClient(predict_app) as client:
            response = client.get('/health')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), 200)


class TestPredict(TestCase):
    def test_correct(self):
        json_data = deepcopy(DATA)
        with TestClient(predict_app) as client:
            response = client.post('/predict', json=json_data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'preds': [1, 1, 1, 1, 1]})

    def test_missed_column(self):
        json_data = deepcopy(DATA)
        # pydantic will not validate this json
        del json_data['age']
        with TestClient(predict_app) as client2:
            response = client2.post('/predict', json=json_data)
            self.assertEqual(response.status_code, 422)

    def test_redundant_column(self):
        json_data = deepcopy(DATA)
        # pydantic will not consider this column
        json_data['xxx'] = 1
        with TestClient(predict_app) as client2:
            response = client2.post('/predict', json=json_data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'preds': [1, 1, 1, 1, 1]})


if __name__ == '__main__':
    unittest.main()
