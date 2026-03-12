#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 9: Pipeline Pattern Model Serving
"""
import pandas as pd

df = pd.DataFrame({'X': [5, 5, 5, 3, 4, 5], 'Y': [2, 2, 2, 1, 1, 1]})
df.to_csv('../airflow/DAGs/Stages/Data/Source 2.csv')
