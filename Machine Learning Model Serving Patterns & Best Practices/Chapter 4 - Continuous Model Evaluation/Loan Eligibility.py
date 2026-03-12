#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 4: Continuous Model Evaluation
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

predictions = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Jan_actual = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Feb_actual = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Mar_actual = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
Apr_actual = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
May_actual = [0, 0, 0, 1, 1, 1, 0, 1, 0, 0]
June_actual = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
July_actual = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

accuracy = []
for actual in [Jan_actual, Feb_actual, Mar_actual, Apr_actual, May_actual, June_actual, July_actual]:
    accuracy.append(accuracy_score(actual, predictions))

plt.plot(accuracy)
plt.xlabel('Month')
plt.ylabel('Accuracy')