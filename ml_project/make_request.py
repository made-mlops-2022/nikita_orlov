import requests
import pandas as pd
import click

SERVER = 'http://0.0.0.0:8000/predict'


@click.command()
@click.argument('data_path')
def predict(data_path: str):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print({'error': 'File not found'})
        return
    data_json = {k: list(v.values()) for k, v in df.to_dict().items()}
    response = requests.post(SERVER, json=data_json)
    if response.status_code == 200:
        print(response.json())
        return
    print({'error': response.json()})


if __name__ == '__main__':
    predict()
