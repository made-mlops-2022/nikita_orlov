# mlops_made
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
~~~# mlops homework
