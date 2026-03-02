import pandas as pd
import numpy as np

def load_data(data_location):
    energy_ts_df = pd.read_csv(data_location, parse_dates = ['Timestamp'])
    energy_ts_df.index = pd.date_range(start = min(energy_ts_df['Timestamp']), end = max(energy_ts_df['Timestamp']), freq = 'h')
    energy_ts_df = energy_ts_df.drop(['Timestamp', 'Temperature'], axis =  1)
    return energy_ts_df

def MAPE(actual, predicted):
    return (np.abs(actual - predicted)/actual).mean()