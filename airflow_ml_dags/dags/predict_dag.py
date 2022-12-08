from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from docker.types import Mount


FEATURES_CONFIG = '/feature_params.yaml'
PROJECT_PATH = '/home/lolvista/MADE/mlops_course/mlops_made_for_org/airflow_ml_dags'
PRODUCTION_MODEL_DAY = Variable.get('PRODUCTION_MODEL_DAY')
DOCKER_ARGS = dict(network_mode='bridge',
                   do_xcom_push=False,
                   mount_tmp_dir=False,
                   mounts=[
                       Mount(source=f'{PROJECT_PATH}/data', target='/data', type='bind'),
                       Mount(source=f'{PROJECT_PATH}/logs', target='/logs', type='bind')])

with DAG(dag_id='predict_dag',
         schedule_interval='@daily',
         start_date=days_ago(10)) as dag:

    start = EmptyOperator(task_id='start')

    predict_data = DockerOperator(task_id='docker_predict_data',
                                  image='predict_data',
                                  command='--data-dir /data/raw/{{ ds }} '
                                          '--prediction-dir /data/prediction/{{ ds }} '
                                          f'--transformer-dir /data/transformers/{PRODUCTION_MODEL_DAY} '
                                          f'--model-dir /data/models/{PRODUCTION_MODEL_DAY}',
                                  **DOCKER_ARGS)

    end = EmptyOperator(task_id='end')

    start >> predict_data >> end
