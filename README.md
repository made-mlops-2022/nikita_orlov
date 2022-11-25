# mlops_made
## Training (HW1)
### Подготовка окружения
Создание виртуального окружения (команды запускаются из `ml_project` директории, unix-like системах):
~~~
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
~~~
### Посмотреть EDA 

В `ml_project` вызывается команда, сам ноутбук лежит в директории `notebooks/EDA.ipynb`:
```
jupyter-notebook
```

### Обучение модели:

В `ml_project` вызывается команда (`config_path` - путь до `.yaml` файла):
~~~
python train.py config_path
python train.py configs/log_reg_no_regularization.yaml
~~~

### Предсказание модели:

В `ml_project` вызывается команда (`model_path` - путь до модели, `data_path` - путь до csv-файла(важно наличие столбцов как при обучении),
`output_path` - путь для файла с предсказаниями)
~~~
python predict.py model_path data_path output_path
python predict.py models/model.pkl data/test/data.csv prediction.csv
~~~
## Deployment (HW2)
### Поднять сервис локально:
Команды запускаются в директории `ml_project`:
~~~
uvicorn app:predict_app --reload
~~~
### Прогнать запросы к сервису:
`data_path` - путь до данных (csv), `num_port` - порт сервиса (обычно `8000`)
~~~
python make_request.py data_path num_port
~~~
### Cборка и запуск докер контейнера
~~~
docker build -t nikovtb/prediction_service .
docker run -it -p 80:80 nikovtb/prediction_service
~~~
убедиться в корректности работы сервиса:
`data_path` - путь до данных (csv), `num_port` - порт сервиса (`80`)
~~~
python make_request.py data_path num_port
~~~
### Контейнер залит в репозиторий Docker-а
можно запускать как и локально
~~~
docker run -it -p 80:80 nikovtb/prediction_service
~~~