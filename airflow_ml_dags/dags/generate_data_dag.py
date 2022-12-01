from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

PROJECT_PATH = '/home/lolvista/MADE/mlops_course/mlops_made_for_org/airflow_ml_dags'
DATASET_LEN = 200


def _generate_data(ds):
    print(f'generate {ds}')


with DAG('generate_data_dag',
         schedule_interval='@daily',
         start_date=days_ago(50),
         ) as dag:
    start = EmptyOperator(task_id='start')

    generate_data = DockerOperator(
        image='generate_data',
        command='--output-dir /data/raw/{{ ds }}',
        network_mode='bridge',
        task_id='docker_generate_data',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=f'{PROJECT_PATH}/data',
                      target='/data',
                      type='bind'),
                Mount(source=f'{PROJECT_PATH}/logs',
                      target='/logs',
                      type='bind')]
    )

    end = EmptyOperator(task_id='end')

    start >> generate_data >> end
