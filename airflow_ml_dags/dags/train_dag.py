from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

FEATURES_CONFIG = '/feature_params.yaml'
PROJECT_PATH = '/home/lolvista/MADE/mlops_course/mlops_made_for_org/airflow_ml_dags'

DOCKER_ARGS = dict(network_mode='bridge',
                   do_xcom_push=False,
                   mount_tmp_dir=False,
                   mounts=[
                       Mount(source=f'{PROJECT_PATH}/data', target='/data', type='bind'),
                       Mount(source=f'{PROJECT_PATH}/logs', target='/logs', type='bind')])

with DAG('train_dag',
         schedule_interval='@weekly',
         start_date=days_ago(50)) as dag:
    train_test_split = DockerOperator(task_id='docker_train_test_split_data',
                                      image='split_data',
                                      command='--input-dir /data/raw/{{ ds }} '
                                              '--config-path /splitting_params.yaml '
                                              '--train-dir /data/train_raw/{{ ds }} '
                                              '--test-dir /data/test_raw/{{ ds }}',
                                      **DOCKER_ARGS
                                      )

    prepare_data = DockerOperator(task_id='docker_prepare_data',
                                  image='prepare_data',
                                  command='--data-dir /data/train_raw/{{ ds }} '
                                          '--config-path /feature_params.yaml '
                                          '--output-dir /data/train_processed/{{ ds }} '
                                          '--transformer-dir /data/transformers/{{ ds }}',
                                  **DOCKER_ARGS)

    train_model = DockerOperator(task_id='docker_train_model',
                                 image='train_model',
                                 command='--data-dir /data/train_processed/{{ ds }} '
                                         '--config-path /training_config.yaml '
                                         '--model-dir /data/models/{{ ds }} '
                                         '--metric-dir /data/metrics/{{ ds }}',
                                 **DOCKER_ARGS)

    validate_model = DockerOperator(task_id='docker_validate_model',
                                    image='validate_model',
                                    command='--data-dir /data/test_raw/{{ ds }} '
                                            '--transformer-dir /data/transformers/{{ ds }} '
                                            '--model-dir /data/models/{{ ds }} '
                                            '--metric-dir /data/metrics/{{ ds }}',
                                    **DOCKER_ARGS)

    train_test_split >> prepare_data >> train_model >> validate_model
