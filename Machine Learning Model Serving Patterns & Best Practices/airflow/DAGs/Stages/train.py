#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 9: Pipeline Pattern Model Serving
"""
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/Users/edwinjosiahgoh95/airflow/DAGs/Stages/Data/Combined.csv')
logreg_model = LogisticRegression(random_state = 0).fit(df['X'].to_numpy().reshape(-1, 1), df['Y'])

with open('/Users/edwinjosiahgoh95/airflow/DAGs/Stages/Demo Model.pkl', 'wb') as file:
    pickle.dump(logreg_model, file)