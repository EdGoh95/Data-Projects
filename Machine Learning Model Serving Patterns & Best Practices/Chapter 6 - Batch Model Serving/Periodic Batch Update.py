#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 6: Batch Model Serving
"""
import random
import pandas as pd

random_scores = []
for k in range(0, 5):
    x = round(random.random(), 2)
    random_scores.append(x)

df = pd.DataFrame({'Product': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
                   'Score': random_scores})
df.to_csv('Predictions.csv')