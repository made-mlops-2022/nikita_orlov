import requests
import pandas as pd
import click

SERVER = 'http://127.0.0.1:80/predict'


@click.command()
@click.argument('data_path')
@click.argument('port')
def predict(data_path: str, port=8000):
    server = f'http://127.0.0.1:{port}/predict'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print({'error': 'File not found'})
        return
    data_json = {k: list(v.values()) for k, v in df.to_dict().items()}
    response = requests.post(server, json=data_json)
    if response.status_code == 200:
        print(response.json())
        return
    print({'error': response.json()})


if __name__ == '__main__':
    predict()
