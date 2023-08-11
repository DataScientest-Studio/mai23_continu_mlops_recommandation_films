from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
import datetime
import wget
import os

my_dag = DAG(
    dag_id = 'my_dag_recommendation',
    description = 'DAG permettant de mettre à jour toutes les semaines les données IMDB',
    schedule_interval = '0 0 * * 1',
    tags = ['recommendation_system'],
    default_args = {
        'owner' : 'anthony',
        'start_date' : datetime.datetime(2023,8,11)
    }

)

def refresh_IMDB_dataset():
    

    # supprime les anciens fichiers IMDB
    os.remove('./data/IMDB/title.basics.tsv.gz')
    os.remove('./data/IMDB/title.crew.tsv.gz')
    os.remove('./data/IMDB/title.principals.tsv.gz')
    os.remove('./data/IMDB/title.ratings.tsv.gz')

    list_url = ['https://datasets.imdbws.com/title.principals.tsv.gz','https://datasets.imdbws.com/title.ratings.tsv.gz',
                    'https://datasets.imdbws.com/title.crew.tsv.gz','https://datasets.imdbws.com/title.basics.tsv.gz']
        
    #télécharger les fichiers IMDB les plus récents
    for url in list_url:
        wget.download(url, out = '../data/IMDB/')


my_refresh_task = PythonOperator(
    task_id = 'refresh_IMDB_dataset',
    dag = my_dag,
    python_callable = refresh_IMDB_dataset,
    retries = 2,
    retry_delay = datetime.timedelta(seconds = 600),
    email_on_retry = True,
    email = ['anthonyferre35770@gmail.com']
)

my_sensor = FileSensor(
    task_id="check_imdb_dataset",
    fs_conn_id="my_filesystem_connection",
    filepath="title.basics.tsv.gz",
    poke_interval=30,
    dag=my_dag,
    timeout=5 * 30,
    mode='reschedule'
)

my_task = BashOperator(
    task_id="print_file_content",
    bash_command="cat ../data/IMDB/title.basics.tsv.gz",
    dag=my_dag
)

my_sensor >> my_refresh_task
my_task >> my_sensor