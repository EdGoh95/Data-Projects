#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 6 - Experimentation in action: Testing and evaluating an ML project
"""
import numpy as np
import pandas as pd
from helper_functions import get_airport_data, smoothed_time_series_plots, apply_index_frequency,\
    generate_splits, plot_predictions, stationary_tests, Holt_Winters_Exponential_Smoothing
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA

#%% Testing Different ML Approaches
EWR_airport_data = get_airport_data('EWR', '../TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
EWR_reference = smoothed_time_series_plots(
    EWR_airport_data['International Passengers'], 'Newark International Airport',
    'Newark_International_Smoothed_Plot.svg', 12, exponential_alpha = 0.5)

# Rapid Testing Of The Vector AutoRegression (VAR) Model Approach
JFK_airport_data = get_airport_data('JFK', '../TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
JFK_airport_data = apply_index_frequency(JFK_airport_data, 'MS')
JFK_training_dataset, JFK_testing_dataset = generate_splits(JFK_airport_data, '2006-07-08')
VAR_model_JFK = VAR(JFK_training_dataset[['Domestic Passengers', 'International Passengers']])
VAR_model_JFK.select_order(12)
VAR_model_JFK_fit = VAR_model_JFK.fit(12)
VAR_prediction_JFK = VAR_model_JFK_fit.forecast(JFK_testing_dataset[
    ['Domestic Passengers', 'International Passengers']].values[-VAR_model_JFK_fit.k_ar:],
    JFK_testing_dataset.index.size)
VAR_prediction_domestic_passengers_JFK = pd.Series(
    np.asarray(list(zip(*VAR_prediction_JFK))[0], dtype = np.float32), index = JFK_testing_dataset.index)
VAR_prediction_international_passengers_JFK = pd.Series(
    np.asarray(list(zip(*VAR_prediction_JFK))[1], dtype = np.float32), index = JFK_testing_dataset.index)
VAR_prediction_score_domestic = plot_predictions(
    JFK_testing_dataset['Domestic Passengers'], VAR_prediction_domestic_passengers_JFK,
    'JFK (VAR Model)', 'Domestic Passengers', 'JFK Domestic Passengers (VAR Model) With Lag = 12.svg')
VAR_prediction_score_international = plot_predictions(
    JFK_testing_dataset['International Passengers'], VAR_prediction_international_passengers_JFK,
    'JFK (VAR Model)', 'International Passengers',
    'JFK International Passengers (VAR Model) With Lag = 12.svg')

# Stationarity-Adjusted Predictions With A VAR Model
JFK_stationary = JFK_airport_data
JFK_stationary['Domestic Differenced'] = np.log(JFK_stationary['Domestic Passengers']).diff()
JFK_stationary['International Differenced'] = np.log(JFK_stationary['International Passengers']).diff()
JFK_stationary = JFK_stationary.dropna()
JFK_stationary_training_dataset, JFK_stationary_testing_dataset = generate_splits(JFK_stationary,
                                                                                  '2006-07-08')
VAR_model_JFK_stationary = VAR(JFK_stationary_training_dataset[['Domestic Differenced',
                                                                'International Differenced']])
VAR_model_JFK_stationary.select_order(6)
VAR_model_JFK_stationary_fit = VAR_model_JFK_stationary.fit(12)
VAR_prediction_stationary_JFK = VAR_model_JFK_stationary_fit.forecast(JFK_stationary_testing_dataset[
    ['Domestic Differenced', 'International Differenced']].values[-VAR_model_JFK_stationary_fit.k_ar:],
    JFK_stationary_testing_dataset.index.size)
VAR_stationary_prediction_domestic_JFK = pd.Series(
    np.asarray(list(zip(*VAR_prediction_stationary_JFK))[0], dtype = np.float32),
    index = JFK_stationary_testing_dataset.index)
VAR_prediction_domestic_JFK_expanded =  np.exp(VAR_stationary_prediction_domestic_JFK.cumsum())\
    * JFK_stationary_testing_dataset['Domestic Passengers'][0]
VAR_stationary_prediction_international_JFK = pd.Series(
    np.asarray(list(zip(*VAR_prediction_stationary_JFK))[1], dtype = np.float32),
    index = JFK_stationary_testing_dataset.index)
VAR_prediction_international_JFK_expanded =  np.exp(VAR_stationary_prediction_international_JFK.cumsum())\
    * JFK_stationary_testing_dataset['International Passengers'][0]
VAR_stationary_prediction_score_domestic = plot_predictions(
    JFK_stationary_testing_dataset['Domestic Passengers'],
    VAR_prediction_domestic_JFK_expanded, 'JFK (VAR Model - Differenced)',
    'Domestic Passengers Differenced',  'JFK Domestic Passengers (VAR Model - Differenced) With Lag = 12.svg')
VAR_stationary_prediction_score_international = plot_predictions(
    JFK_stationary_testing_dataset['International Passengers'],
    VAR_prediction_international_JFK_expanded, 'JFK (VAR Model - Differenced)',
    'International Passengers Differenced', 'JFK International Passengers (VAR Model - Differenced) With Lag = 12.svg')

# Stationarity Testing Using ACF and PACF Plots
JFK_stationary_time_series_plots = stationary_tests(
    JFK_stationary, 'Domestic Differenced',  'JFK', 12,
    'LogDiff Of JFK Domestic Passengers Time Series Plots.svg', 48)

# Rapid Testing Of ARIMA Model Approach
ARIMA_model_JFK_domestic = ARIMA(JFK_training_dataset['Domestic Passengers'], order = (43, 1, 1),
                                 enforce_stationarity = False, trend = 'n')
ARIMA_model_JFK_international = ARIMA(JFK_training_dataset['International Passengers'],
                                      order = (42, 1, 1), enforce_stationarity = False, trend = 'n')
ARIMA_fit_JFK_domestic = ARIMA_model_JFK_domestic.fit()
ARIMA_fit_JFK_international = ARIMA_model_JFK_international.fit()
ARIMA_prediction_JFK_domestic = ARIMA_fit_JFK_domestic.predict(JFK_testing_dataset.index[0],
                                                               JFK_testing_dataset.index[-1])
ARIMA_prediction_JFK_international = ARIMA_fit_JFK_international.predict(JFK_testing_dataset.index[0],
                                                                         JFK_testing_dataset.index[-1])
ARIMA_prediction_score_JFK_domestic = plot_predictions(
    JFK_testing_dataset['Domestic Passengers'], ARIMA_prediction_JFK_domestic, 'JFK (ARIMA Model)',
    'Domestic Passengers', 'JFK Domestic Passengers - ARIMA(42,1,1).svg')
ARIMA_prediction_score_JFK_international = plot_predictions(
    JFK_testing_dataset['International Passengers'], ARIMA_prediction_JFK_international, 'JFK (ARIMA Model)',
    'International Passengers', 'JFK International Passengers - ARIMA(42,1,1).svg')

# Rapid Testing Of Holt-Winters Exponential Smoothing Algorithm Approach
prediction_JFK_domestic = Holt_Winters_Exponential_Smoothing(
    JFK_training_dataset['Domestic Passengers'], JFK_testing_dataset['Domestic Passengers'], 'add',
    'add', 48, True, 0.9, 0.5)
prediction_JFK_international = Holt_Winters_Exponential_Smoothing(
    JFK_training_dataset['International Passengers'], JFK_testing_dataset['International Passengers'], 'add',
    'add', 48, True, 0.1, 1.0)
Holt_Winters_prediction_JFK_domestic = plot_predictions(
    JFK_testing_dataset['Domestic Passengers'], prediction_JFK_domestic['Forecast'],
    'JFK (Holt-Winters Exponential Smoothing)', 'Domestic Passengers',
    'JFK Domestic Passengers (Holt-Winters Exponential Smoothing).svg')
Holt_Winters_prediction_JFK_international = plot_predictions(
    JFK_testing_dataset['International Passengers'], prediction_JFK_international['Forecast'],
    'JFK (Holt-Winters Exponential Smoothing)', 'International Passengers',
    'JFK International Passengers (Holt-Winters Exponential Smoothing).svg')