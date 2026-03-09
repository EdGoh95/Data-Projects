#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 11 - Model measurement and why it's important
"""
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from statsmodels.stats.power import tt_ind_solve_power
from datetime import datetime, timedelta, date
from statistics_utilities import plot_comparison_series_df, plot_anova, plot_tukey, plot_coupon_usage

x_effects = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
sample_sizes = [tt_ind_solve_power(x, None, 0.2, 0.8, 1, 'two-sided') for x in x_effects]
sample_sizes_low_alpha = [tt_ind_solve_power(x, None, 0.01, 0.8, 1, 'two-sided') for x in x_effects]

with plt.style.context('seaborn-v0_8'):
    plt.figure(figsize = (10, 8))
    plt.plot(x_effects, sample_sizes, color = 'blue')
    plt.scatter(x_effects, sample_sizes, label = r'$\alpha=0.2$', color = 'blue')
    plt.plot(x_effects, sample_sizes_low_alpha, color = 'red')
    plt.scatter(x_effects, sample_sizes_low_alpha, label = r'$\alpha=0.01$', color = 'red')
    plt.title('Relationship Between The Mean Sample Difference And Sample Size - An Example')
    plt.xlabel('Sample Mean Difference')
    plt.ylabel('Required Sample Size')
    plt.yscale('log')
    plt.legend(loc = 'best')
    for index, value in enumerate(sample_sizes):
        plt.annotate(np.round(value, 0).astype(int), (x_effects[index] + 0.01, value))
    for index, value in enumerate(sample_sizes_low_alpha):
        plt.annotate(np.round(value, 0).astype(int), (x_effects[index] + 0.01, value))
    plt.tight_layout()
    plt.savefig('Sample Sizes And Mean Sample Difference.jpg', dpi = 600)

raw_typical_revenue = np.abs(np.random.standard_cauchy(100000))
typical_revenue = raw_typical_revenue[(raw_typical_revenue > 1) & (raw_typical_revenue < 20)] * 20

with plt.style.context('seaborn-v0_8'):
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111)
    ax.hist(typical_revenue, linewidth = 1.5, edgecolor = 'white', bins = 50)
    ax.set_title('Revenue Distribution Per Order')
    ax.set_xlabel('Spend Amount')
    ax.set_ylabel('Number Of Customers Last Week')
    plt.tight_layout()
    plt.savefig('A Typical Revenue Distribution.jpg', dpi = 600)

#%% Generate Data For A/B Testing
def filter_start(data, days_forward = 30):
    return data[data['Date'] < min(data['Date']) + timedelta(days = days_forward)]

def generate_series(base_start, base_end, variance_mu, variance_sigma, group_name, data_size = 260):
    TestSeries = namedtuple('TestSeries', 'name, data, start, stop, mu, sigma')
    noise_factor = np.random.normal(variance_mu, variance_sigma, data_size)
    generated = np.linspace(start = base_start, stop = base_end, num = data_size) + noise_factor
    return TestSeries(group_name, [0.0 if x < 0.0 else x for x in generated], base_start, base_end,
                      variance_mu, variance_sigma)

def generate_series_df(config, date_range):
    series = []
    for k, v in config.items():
        for pk, pv in v.items():
            series.append(generate_series(
                pv['start'], pv['stop'], pv['mu'], pv['sigma'], '{} {}'.format(pk, k)))
    series_df = pd.DataFrame([s.data for s in series]).T
    series_df.columns = [s.name for s in series]
    series_df['Date'] = date_range
    return series_df

def generate_melted_series_df(series_data, date_range, date_filtering = 260):
    series_df = generate_series_df(series_data, date_range)
    melted_df = pd.melt(series_df.reset_index(), id_vars = 'Date', value_vars = series_df.columns)
    melted_df.columns = ['Date', 'Test', 'Sales']
    return melted_df[melted_df['Date'] > max(melted_df['Date']) - timedelta(days = date_filtering)]

def generate_augmented_series_df(series_data, date_range):
    series_df = generate_series_df(series_data, date_range)
    for sales in ['High', 'Medium', 'Low']:
        series_df['{} Value Sales'.format(sales)] = (series_df['Control {} Value'.format(sales)] +
                                                     series_df['Test {} Value'.format(sales)])
    series_df['Control Sales'] = (series_df['Control High Value'] + series_df['Control Medium Value'] +
                                  series_df['Control Low Value'])
    series_df['Test Sales'] = (series_df['Test High Value'] + series_df['Test Medium Value'] +
                               series_df['Test Low Value'])
    series_df['Total Sales'] = series_df['Control Sales'] + series_df['Test Sales']
    return series_df

config = {'High Value': {'Control': {'start': 5000, 'stop': 145000, 'mu': 6000, 'sigma': 4500},
                         'Test': {'start': 5000, 'stop': 160000, 'mu': 8000, 'sigma': 4500}},
          'Medium Value': {'Control': {'start': 6000, 'stop': 105000, 'mu': 2000, 'sigma': 4000},
                           'Test': {'start': 6000, 'stop': 105000, 'mu': 3000, 'sigma': 6000}},
          'Low Value': {'Control': {'start': 10000, 'stop': 50000, 'mu': 1000, 'sigma': 5000},
                        'Test': {'start': 10000, 'stop': 50000, 'mu': 1750, 'sigma': 8000}}}

START_DATE = datetime(2020, 12, 1)
date_range = np.arange(START_DATE, START_DATE + timedelta(days = 260), timedelta(days = 1)).astype(date)
series = generate_augmented_series_df(config, date_range)
first_150_days = filter_start(series, 150)
plot_comparison_series_df(first_150_days['Date'], first_150_days['Control High Value'],
                          first_150_days['Test High Value'],
                          'First 150 Days Of Sales For High Value Customers', (8, 8))

with plt.style.context('seaborn-v0_8'):
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111)
    ax.hist(first_150_days['Test High Value'], linewidth = 1.5, edgecolor = 'white', bins = 25)
    ax.set_title('High Value Customer Revenue - Test Group')
    ax.set_xlabel('Revenue')
    ax.set_ylabel('Days')
    plt.tight_layout()
    plt.savefig('Revenue Distribution For High Value Customers - Test Group.jpg', dpi = 600)

config_stationary = {
    'High Value': {'Control': {'start': 150000, 'stop': 151000, 'mu': 6000, 'sigma': 4500},
                   'Test': {'start': 185000, 'stop': 186000, 'mu': 8000, 'sigma': 4500}},
    'Medium Value': {'Control': {'start': 100000, 'stop': 100500, 'mu': 2000, 'sigma': 4000},
                     'Test': {'start': 100000, 'stop': 100500, 'mu': 2000, 'sigma': 8000}},
    'Low Value': {'Control': {'start': 50000, 'stop': 50500, 'mu': 1000, 'sigma': 5000},
                  'Test': {'start': 52000, 'stop': 52500, 'mu': 1750, 'sigma': 8000}}}
config_stationary_anova = {'High Value': config_stationary['High Value']}

stationary_series_anova = generate_melted_series_df(config_stationary_anova, date_range, 120)
plot_anova(stationary_series_anova, 'Stationary Time Series For The First 120 Days (High Value Customers)',
           (9, 9))

with plt.style.context('seaborn-v0_8'):
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_subplot(111)
    ax.hist(generate_series_df(config_stationary_anova, date_range)['Test High Value'],
            linewidth = 1.5, edgecolor = 'white', bins = 25)
    ax.set_title('High Value Customer Revenue - Test Group (Stationary)')
    ax.set_xlabel('Revenue')
    ax.set_ylabel('Days')
    plt.tight_layout()
    plt.savefig('Revenue Distribution For High Value Customers - Test Group (Stationary).jpg', dpi = 600)

stationary_series_melted = generate_melted_series_df(config_stationary, date_range, 120)
plot_tukey(stationary_series_melted, 'Stationary Time Series For The First 120 Days')

coupons_date_range = np.arange(START_DATE, START_DATE + timedelta(days = 50), timedelta(days = 1)).astype(date)
control_series = np.random.uniform(low = 1000, high = 10000, size = 50).astype(int) + \
    np.linspace(start = 0, stop = 5000, num = 50).astype(int)
control_unused = [25000 - c for c in control_series]
test_series = np.random.uniform(low = 2000, high = 10000, size = 50).astype(int) + \
    np.linspace(start = 0, stop = 10000, num = 50).astype(int)
test_unused = [25000 - c for c in test_series]
plot_coupon_usage(test_series, test_unused, control_series, control_unused, coupons_date_range,
                  '50 Days')