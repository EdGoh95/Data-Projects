#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

def logistic_map(x, recurrence):
    '''
    Function for recursing over prior values
    '''
    return x * recurrence * (1 - x)

def log_map(n, x, r, collection = None):
    '''
    Tail-recursive function for generating the series by applying the logistic map equation over
    each previous value
    '''
    if collection is None:
        collection = []
    calculated_value = logistic_map(x, r)
    collection.append(calculated_value)
    if n > 0:
        log_map(n-1, calculated_value, r, collection)
    return np.array(collection[:n])

def generate_log_map_and_plot(iterations, recurrence, start):
    '''
    Function for generating the series and a plot to show what the particular recurrence value does
    to the series
    '''
    map_series = log_map(iterations, start, recurrence)
    MapData = namedtuple('MapData', 'series, plot')
    with plt.style.context(style = 'seaborn-v0_8'):
        fig = plt.figure(figsize = (15, 8))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(iterations), map_series)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Logistic Map Values')
        ax.set_title('Logistic Map With Recurrence Of {}'.format(recurrence))
    return MapData(map_series, fig)

def analyze_series(series):
    BasicStats = namedtuple('BasicStats', 'minimum, mean, maximum')
    Variation = namedtuple('Variation', 'std_dev, variance')
    Quantiles = namedtuple('Quantiles', 'p5, q1, median, q3, p95')
    StatAnalysis = namedtuple('StatAnalysis', ['basic_stats', 'variation', 'quantiles'])
    minimum = np.min(series)
    mean = np.average(series)
    maximum = np.max(series)
    q1 = np.quantile(series, 0.25)
    median = np.quantile(series, 0.5)
    q3 = np.quantile(series, 0.75)
    p5, p95 = np.percentile(series, [5, 95])
    std_dev = np.std(series)
    variance = np.var(series)
    return StatAnalysis(BasicStats(minimum, mean, maximum), Variation(std_dev, variance),
                        Quantiles(p5, q1, median, q3, p95))

log_map_chaos_series = generate_log_map_and_plot(1000, 3.7223976, 0.5)
chaos_series_statistics = analyze_series(log_map_chaos_series.series)
print(chaos_series_statistics.variation.std_dev)