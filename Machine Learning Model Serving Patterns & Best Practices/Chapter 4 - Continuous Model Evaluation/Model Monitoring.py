#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 4: Continuous Model Evaluation
"""
import numpy as np
import matplotlib.pyplot as plt

MSE = np.array([0, 4000, 7000, 11000, 15000, 23000])
RMSE = np.sqrt(MSE)
MAE = np.array([0, 200, 300, 500, 700, 1000])
MAPE = np.array([0, 3, 8, 12, 20, 30])

fig, axes = plt.subplots(2, 2)
fig.suptitle('Model Monitoring Dashboard', size = 25, y = 0.95)

axes[0, 0].plot(MSE)
axes[0, 0].axhline(y = 5000, color = 'r', linestyle = '--', label = 'MSE Threshold')
axes[0, 0].set(xlabel = 'Month', ylabel = 'MSE')
axes[0, 0].legend()

axes[0, 1].plot(RMSE)
axes[0, 1].set(xlabel = 'Month', ylabel = 'RMSE')

axes[1, 0].plot(MAE)
axes[1, 0].set(xlabel = 'Month', ylabel = 'MAE')

axes[1, 1].plot(MAPE)
axes[1, 1].set(xlabel = 'Month', ylabel = 'MAPE')