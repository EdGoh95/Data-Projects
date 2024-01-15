#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 5 - Experimentation in action: Planning and researching an ML project
"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from helper_functions import apply_index_frequency, get_airport_data, generate_outlier_plots, stationary_tests

#%% Exploratory Data Analysis (EDA)
airport_traffic_df = pd.read_csv('TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
airport_traffic_df = airport_traffic_df.copy(deep = False)
airport_traffic_df['Month'] = pd.to_datetime(airport_traffic_df['Month'], format = '%b').dt.month
airport_traffic_df.loc[:, 'Day'] = 1
airport_traffic_df['date'] = pd.to_datetime(airport_traffic_df[['Year', 'Month', 'Day']])
JFK_airport_traffic_df = airport_traffic_df[airport_traffic_df['Airport Code'] == 'JFK']
JFK_airport_traffic_df = JFK_airport_traffic_df.sort_values('date', ascending = True)
JFK_airport_traffic_df.set_index('date', inplace = True)
plt.figure()
plt.plot(JFK_airport_traffic_df['Domestic Passengers'])

# Generating A Moving Average Trend Based On Smoothing Period Of 1 Year (With A 2-Sigma Error)
JFK_moving_average = JFK_airport_traffic_df['Domestic Passengers'].rolling(12, center = False).mean()
JFK_moving_std = JFK_airport_traffic_df['Domestic Passengers'].rolling(12, center = False).std()
plt.figure()
plt.plot(JFK_airport_traffic_df['Domestic Passengers'], label = 'Monthly Passenger Count')
plt.plot(JFK_moving_average, color = 'red', label = 'Moving Average')
plt.plot(JFK_moving_average + (2 * JFK_moving_std), color = 'green', linestyle = '-.',
         label = r'Moving 2$\sigma$ Error')
plt.plot(JFK_moving_average - (2 * JFK_moving_std), color = 'green', linestyle = '-.')
plt.title('JFK Passenger Count By Month')
plt.legend(loc = 'best')

# Stationary Test Of The Time Series
JFK_dickey_fuller_test = adfuller(JFK_airport_traffic_df['Domestic Passengers'], autolag = 'AIC')
JFK_test_report = JFK_dickey_fuller_test[:4] + (
    ('Non-' if JFK_dickey_fuller_test[:4][1] > 0.05 else '') + 'stationary', )
JFK_stationary_test_report_df = pd.Series(JFK_test_report,
                                          index = ['Test Statistic', 'p-value', 'Number of Lags',
                                                   'Number Of Observations', 'Stationary Test'])
for k, v in JFK_dickey_fuller_test[4].items():
    JFK_stationary_test_report_df['Critical Value (%s)' % k] = v
print(JFK_stationary_test_report_df)

# Trend Decomposition For Seasonality
trend_decomposition = seasonal_decompose(JFK_airport_traffic_df['Domestic Passengers'], period = 12)
trend_plot = trend_decomposition.plot()
trend_plot.set_size_inches(15, 8)
plt.savefig('Trend Decomposition Of JFK Domestic Passenger Count.svg', format = 'svg', dpi = 600)

# Time-Series Differencing To Identify Outliers
JFK_airport_traffic_df['Log Domestic Passengers'] = np.log(JFK_airport_traffic_df['Domestic Passengers'])
JFK_airport_traffic_df['DiffLog Domestic Passengers By Month'] = JFK_airport_traffic_df['Log Domestic Passengers'].diff(1)
JFK_airport_traffic_df['DiffLog Domestic Passengers By Year'] = JFK_airport_traffic_df['Log Domestic Passengers'].diff(12)
fig, axes = plt.subplots(3, 1, figsize = (15, 8.5), constrained_layout = True)
boundary1 = datetime.datetime.strptime('2001-07-01', '%Y-%m-%d')
boundary2 = datetime.datetime.strptime('2001-11-01', '%Y-%m-%d')
axes[0].plot(JFK_airport_traffic_df['Domestic Passengers'], '-', label = 'Domestic Passengers')
axes[0].set(title = 'JFK Domestic Passengers')
axes[0].axvline(boundary1, 0, 2.5e6, color = 'red', linestyle = '--', label = r'Sept 11$^{th}$, 2001')
axes[0].axvline(boundary2, 0, 2.5e6, color = 'red', linestyle = '--')
axes[0].legend(loc = 'upper left')
axes[1].plot(JFK_airport_traffic_df['DiffLog Domestic Passengers By Month'],
             label = 'Monthly Difference In Domestic Passengers')
axes[1].hlines(0, JFK_airport_traffic_df.index[0], JFK_airport_traffic_df.index[-1], 'green')
axes[1].set(title = 'JFK Domestic Passengers Log Diff = 1')
axes[1].axvline(boundary1, 0, 2.5e6, color = 'red', linestyle = '--', label = r'Sept 11$^{th}$, 2001')
axes[1].axvline(boundary2, 0, 2.5e6, color = 'red', linestyle = '--')
axes[1].legend(loc = 'lower left')
axes[2].plot(JFK_airport_traffic_df['DiffLog Domestic Passengers By Year'],
             label = 'Yearly Difference In Domestic Passengers')
axes[2].hlines(0, JFK_airport_traffic_df.index[0], JFK_airport_traffic_df.index[-1], 'green')
axes[2].set(title = 'JFK Domestic Passengers Log Diff = 12')
axes[2].axvline(boundary1, 0, 2.5e6, color = 'red', linestyle = '--', label = r'Sept 11$^{th}$, 2001')
axes[2].axvline(boundary2, 0, 2.5e6, color = 'red', linestyle = '--')
axes[2].legend(loc = 'lower left')
plt.savefig('LogDiff Of JFK Domestic Passenger Count.svg', format = 'svg', dpi = 600)

#%% Using Re-Usable Code From The 'helper_functions.py' Script
JFK_airport_data = get_airport_data('JFK', '../TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
JFK_airport_data = apply_index_frequency(JFK_airport_data, 'MS')
concorde_retirement = generate_outlier_plots(JFK_airport_data, 'JFK', 'International Passengers',
                                             '2003-10-24', 'Corcorde Retirement', 'Retirement Of The Corcorde.svg')

LGA_airport_data = get_airport_data('LGA', '../TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
LGA_airport_data = apply_index_frequency(LGA_airport_data, 'MS')
sep_11_2001 = generate_outlier_plots(LGA_airport_data, 'LGA', 'Domestic Passengers', '2001-09-11',
                                     'Impact of 9/11 On Domestic Passenger Count',
                                     'Impact of Sept 11 2001 On LGA Passenger Count.svg')

# Time Series Decomposition For Newark International Airport (EWR)
EWR_airport_data = get_airport_data('EWR', '../TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
EWR_airport_data = apply_index_frequency(EWR_airport_data, 'MS')
EWR_time_series_plots = stationary_tests(
    EWR_airport_data, 'Domestic Passengers', 'Newark International Airport', 12,
    'Newark International Airport Domestic Passengers Time Series Plots.svg', 48)
