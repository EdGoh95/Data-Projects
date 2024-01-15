#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 7 - Experimentation in action: Moving from prototype to MVP
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, Trials, tpe
from helper_functions import get_airport_data, apply_index_frequency, generate_splits,\
    HWES, HWES_optimization_function, run_hyperparameter_tuning
from pyspark.sql import types, functions, SparkSession

#%% Introduction To HyperOpt (Tree-Structured Parzen Estimators - TPEs)
def objective_function(x):
    func = np.poly1d([1, -3, -88, 112, -5])
    return func(x) * 0.01

trials = Trials()
trial_estimator = fmin(fn = objective_function, space = hp.uniform('x', -12, 12), algo = tpe.suggest,
                       trials = trials, max_evals = 1000)

rng = np.arange(-11.0, 12.0, 0.01)
values = [objective_function(x) for x in rng]
with plt.style.context(style = 'seaborn'):
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    ax.plot(rng, values)
    ax.set_title('Objective Function')
    ax.scatter(x = trial_estimator['x'], y = trials.average_best_error(), marker = 'o', s = 100)
    bbox_text = 'Minimum value calculated using Hyperopt\nx = {}'.format(trial_estimator['x'])
    configuration = dict(xycoords = 'data', textcoords = 'axes fraction',
                          arrowprops = dict(facecolor = 'aqua', shrink = 0.01, connectionstyle = 'angle3'),
                          bbox = dict(boxstyle = 'round,pad=0.5', fc = 'ivory', ec = 'grey', lw = 0.8),
                          ha = 'left', va = 'center', fontsize = 12)
    ax.annotate(bbox_text, xy = (trial_estimator['x'], trials.average_best_error()),
                xytext = (0.3, 0.8), **configuration)
    bbox_text.get_window_extent(fig.canvas.get_renderer())
    fig.tight_layout()

#%% Apply HyperOpt To Optimize The Hyperparameters For The Time-Series Forecasting
hyperopt_search_space = {'model': {'trend': hp.choice('trend', ['add', 'mul']),
                                   'seasonal': hp.choice('seasonal', ['add', 'mul']),
                                   'seasonal_periods': hp.quniform('seasonal_periods', 12, 120, 12),
                                   'damped_trend': hp.choice('damped_trend', [True, False])},
                         'fit': {'smoothing_level': hp.uniform('smoothing_level', 0.01, 0.99),
                                 'smoothing_seasonal': hp.uniform('smoothing_seasonal', 0.01, 0.99),
                                 'damping_trend': hp.uniform('damping_trend', 0.01, 0.99),
                                 'use_brute': hp.choice('use_brute', [True, False]),
                                 'method': hp.choice('method', ['basinhopping', 'L-BFGS-B']),
                                 'remove_bias': hp.choice('remove_bias', [True, False])}}

parameters = {'optimization_function': HWES_optimization_function, 'tuning_space': hyperopt_search_space,
              'forecast_algo': HWES, 'loss_metric': 'BIC', 'hyperopt_algo': tpe.suggest,
              'iterations': 400, 'time_series_name': 'JFK (HyperOpt)', 'value_name': 'Total Passengers',
              'image_name': 'JFK Total Passengers (Hyperparameter Tuning Using HyperOpt).svg'}

JFK_airport_data = get_airport_data('JFK', 'TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
JFK_airport_data = apply_index_frequency(JFK_airport_data, 'MS')
JFK_training_dataset, JFK_testing_dataset = generate_splits(JFK_airport_data, '2014-12-01')
JFK_hyperopt = run_hyperparameter_tuning(JFK_training_dataset['Total Passengers'], 
                                         JFK_testing_dataset['Total Passengers'], parameters)