#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 9: Pipeline Pattern Model Serving
"""
import os

if not os.path.exists('../airflow/DAGs/Stages/Data'):
    os.mkdir('../airflow/DAGs/Stages/Data')
    print('Data directory has been created!')
