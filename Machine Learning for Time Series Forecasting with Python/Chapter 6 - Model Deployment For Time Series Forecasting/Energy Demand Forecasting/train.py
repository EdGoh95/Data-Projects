"""
This script will be given to the Estimator object which is configured in the AML training script.
It is parameterized for training on the 'Energy.csv' dataset
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from helper_functions import load_data,  MAPE

#%% Parameters
COLUMN_OF_INTEREST = 'Load' # Column containing the data that will be forecasted
TRAINING_SAMPLES = 2184
ORDER = (3, 1, 0)
'''
A tuple of 3 non-negative integers specifying the parameters (p, d, q) of an ARIMA model where:
    p - Number of lags in (order of) the AutoRegressive (AR) model
    d - Degree of differencing`
    q - Order of the Moving Average (MA) model
'''
SEASONAL_ORDER = (1, 1, 0, 24)
'''
A tuple of 4 non-negative integers where: 
    - The first 3 integers specify the p, d, q parameters of the ARIMA term of the seasonal component
    - The last integer specifies the number of periods in each season using the parameter, m
'''

#%% Script Arguments
parser = argparse.ArgumentParser(description = 'Input arguments for training the energy demand forecasting model')
parser.add_argument('--data-folder', default = 'Data', type = str, dest = 'data_folder')
parser.add_argument('--filename', default = 'Energy.csv', dest = 'filename')
parser.add_argument('--output', default = 'outputs', type = str, dest = 'output')
args = parser.parse_args()
data_folder = args.data_folder
filename = args.filename
output = args.output

#%% Data Preparation
energy_load_df = load_data(os.path.join(data_folder, filename))
train_df = energy_load_df.iloc[0:TRAINING_SAMPLES]

# Scaling the data to [0, 1]
scaler = MinMaxScaler()
train_df[COLUMN_OF_INTEREST] = scaler.fit_transform(np.array(train_df.loc[:, COLUMN_OF_INTEREST].values).reshape(-1, 1))

#%% Training & Saving The Model
SARIMAX_energy_forecasting_model = SARIMAX(endog = train_df[COLUMN_OF_INTEREST].tolist(), order = ORDER, seasonal_order = SEASONAL_ORDER).fit(disp = False)

os.makedirs(output, exist_ok = True)
with open(os.path.join(output, 'SARIMAX Energy Forecasting.pkl'), 'wb') as model_file:
    pickle.dump(SARIMAX_energy_forecasting_model, model_file)