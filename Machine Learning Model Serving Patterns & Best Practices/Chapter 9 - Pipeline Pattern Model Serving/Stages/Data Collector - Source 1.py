#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 9: Pipeline Pattern Model Serving
"""
import pandas as pd

df = pd.DataFrame({'X': [2, 2, 2, 3, 4, 5], 'Y': [1, 1, 1, 1, 1, 1]})
df.to_csv('/Users/edwinjosiahgoh95/airflow/DAGs/Stages/Data/Source 1.csv')