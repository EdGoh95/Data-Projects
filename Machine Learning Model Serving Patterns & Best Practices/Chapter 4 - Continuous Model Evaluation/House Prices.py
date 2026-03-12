#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 4: Continuous Model Evaluation
"""
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

predictions = [350, 430, 550, 300]
Jan_actual = [350, 430, 550, 300]
Feb_actual = [360, 445, 570, 305]
Mar_actual = [370, 460, 590, 310]
Apr_actual = [380, 475, 610, 315]
May_actual = [390, 500, 630, 325]
June_actual = [410, 515, 650, 330]
July_actual = [430, 530, 670, 340]

MSE = []
for actual in [Jan_actual, Feb_actual, Mar_actual, Apr_actual, May_actual, June_actual, July_actual]:
    MSE.append(mean_squared_error(actual, predictions))

plt.plot(MSE)
plt.xlabel('Month')
plt.ylabel('MSE')