#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 9: Pipeline Pattern Model Serving
"""
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG('Demo-ML-Pipeline',
         default_args = {'depends_on_past': False, 'email': ['<YOUR_EMAIL>'],
                         'email_on_failure': True, 'email_on_retry': False, 'retries': 3,
                         'retry_delay': timedelta(minutes = 5)},
         description = 'A demonstration of how to execute a Machine Learning (ML) pipeline using Apache Airflow',
         schedule_interval = "0/5 * * * *", start_date = datetime(2025, 1, 15), catchup = False,
         tags = ['ML-Pipelines']) as dag:
    init_data_directory = BashOperator(
        task_id = 'Initialize_Data_Directory',
        bash_command = 'python3 "../airflow/DAGs/Stages/Initialize Data Directory.py"')

    data_collection_source1 = BashOperator(
        task_id = 'Source_1_Data_Collection',
        bash_command = 'python3 "../airflow/DAGs/Stages/Data Collector - Source 1.py"')

    data_collection_source2 = BashOperator(
        task_id = 'Source_2_Data_Collection',
        bash_command = 'python3 "../airflow/DAGs/Stages/Data Collector - Source 2.py"')

    data_combiner = BashOperator(
        task_id = 'Combining_Data_From_Different_Sources',
        bash_command = 'python3 "../airflow/DAGs/Stages/Combining Data From Different Sources.py"')

    model_training = BashOperator(
        task_id = 'Model_Training',
        bash_command = 'python3 "../airflow/DAGs/Stages/train.py"')

    init_data_directory >> [data_collection_source1, data_collection_source2] >> data_combiner >> model_training
