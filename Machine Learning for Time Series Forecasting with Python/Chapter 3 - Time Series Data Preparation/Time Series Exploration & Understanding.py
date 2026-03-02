#!/usr/bin/env python3
"""
Machine Learning For Time Series Forecasting With Python Chapter 3: Time Series Data Preparation
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20.0, 12.0]
import matplotlib.dates as dates

energy_ts_df = pd.read_csv('../Data/Energy.csv', parse_dates = ['Timestamp'])
energy_ts_df.index = pd.date_range(
    start = min(energy_ts_df['Timestamp']), end = max(energy_ts_df['Timestamp']), freq = 'H')
energy_ts_df = energy_ts_df.drop('Timestamp', axis =  1)

load_ts_df = energy_ts_df['Load']
load_ts_decomposition_2012 = sm.tsa.seasonal_decompose(load_ts_df.loc['2012-07-01':'2012-12-31'],
                                                      model = 'additive')
load_ts_decomposition_2012.plot()

load_ts_decomposition_full = sm.tsa.seasonal_decompose(load_ts_df, model = 'additive')

fig, axes = plt.subplots()
axes.grid(True)

year = dates.YearLocator(month = 1)
year_format = dates.DateFormatter('%Y')
month = dates.MonthLocator(interval = 1)
month_formatter = dates.DateFormatter('%m')

axes.xaxis.set_minor_locator(month)
axes.xaxis.grid(True, which = 'minor')
axes.xaxis.set_major_locator(year)
axes.xaxis.set_major_formatter(year_format)

plt.plot(load_ts_df.index, load_ts_df, c = 'blue')
plt.plot(load_ts_decomposition_full.trend.index, load_ts_decomposition_full.trend, c = 'white')