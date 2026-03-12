#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 9: Pipeline Pattern Model Serving
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG('My-First-Workflow',
         default_args = {'depends_on_past': False, 'email': ['<YOUR_EMAIL>'],
                         'email_on_failure': True, 'email_on_retry': False, 'retries': 3,
                         'retry_delay': timedelta(minutes = 5)},
         description = 'My first DAG', schedule_interval = timedelta(days = 1),
         start_date = datetime(2025, 1, 15), catchup = False, tags = ['Test']) as dag:
    t1 = BashOperator(task_id = 'Stage_1',
                      bash_command = 'python3 "../airflow/DAGs/Stage 1.py"')

    t2 = BashOperator(task_id = 'Stage_2',
                      bash_command = 'python3 "../airflow/DAGs/Stage 2.py"')

    t1 >> t2
