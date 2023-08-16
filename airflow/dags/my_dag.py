from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator
import datetime
import os
import urllib.request


my_dag = DAG(
    dag_id = 'my_dag_recommendation',
    description = 'DAG permettant de mettre à jour toutes les semaines les données IMDB',
    schedule_interval = '@weekly',
    tags = ['recommendation_system'],
    catchup= True,
    default_args = {
        'owner' : 'airflow',
        'start_date' : datetime.datetime(2023,8,13)
    }

)

def refresh_IMDB_dataset():
    

    # supprime les anciens fichiers IMDB
    os.remove('../../data/IMDB/title.basics.tsv.gz')
    os.remove('../../data/IMDB/title.crew.tsv.gz')
    os.remove('../../data/IMDB/title.principals.tsv.gz')
    os.remove('../../data/IMDB/title.ratings.tsv.gz')

    list_url = ['https://datasets.imdbws.com/title.principals.tsv.gz','https://datasets.imdbws.com/title.ratings.tsv.gz',
                        'https://datasets.imdbws.com/title.crew.tsv.gz','https://datasets.imdbws.com/title.basics.tsv.gz']

    file_name = ['title.principals.tsv.gz','title.ratings.tsv.gz','title.crew.tsv.gz','title.basics.tsv.gz']

    #télécharger les fichiers IMDB les plus récents
    for url in list_url:
        for i in range(4):
            with urllib.request.urlopen(url) as file:
                with open(file = '../../data/IMDB/{}'.format(file_name[i]),mode = 'wb') as new_file:
                    new_file.write(file.read())


my_refresh_task = PythonOperator(
    task_id = 'refresh_IMDB_dataset',
    dag = my_dag,
    python_callable = refresh_IMDB_dataset,
)