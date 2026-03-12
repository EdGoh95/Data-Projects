#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 9: Pipeline Pattern Model Serving
"""
import pandas as pd

df1 = pd.read_csv('../airflow/DAGs/Stages/Data/Source 1.csv')
df2 = pd.read_csv('../airflow/DAGs/Stages/Data/Source 2.csv')
combined_df = pd.concat([df1, df2], ignore_index = True)
print(combined_df)
combined_df.to_csv('../airflow/DAGs/Stages/Data/Combined.csv', index = False)
