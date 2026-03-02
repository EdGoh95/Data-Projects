       #!/usr/bin/env python3
"""
Machine Learning For Time Series Forecasting With Python
Chapter 5: Introduction To Neural Networks For Time Series Forecasting
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('Source Code From GitHub/Notebooks')
from sklearn.preprocessing import MinMaxScaler
from keras import models, layers, callbacks
from common.utils import TimeSeriesTensor, create_evaluation_df

#%% Preparing The Time Series Data For Deep Learning
energy_ts_df = pd.read_csv('Data/Energy.csv', parse_dates = ['Timestamp'])
energy_ts_df.index = pd.date_range(
    start = min(energy_ts_df['Timestamp']), end = max(energy_ts_df['Timestamp']), freq = 'h')
energy_ts_df = energy_ts_df.drop('Timestamp', axis =  1)
load_ts_df = energy_ts_df[['Load']]

validation_timestamp = '2014-09-01 00:00:00'
test_timestamp = '2014-11-01 00:00:00'
load_ts_df.loc[:validation_timestamp].iloc[:-1][['Load']].rename(columns = {'Load':'Train'}).join(
    load_ts_df.loc[validation_timestamp:test_timestamp].iloc[:-1][['Load']].rename(
        columns = {'Load':'Validation'}), how = 'outer').join(
            load_ts_df.loc[test_timestamp:][['Load']].rename(
                columns = {'Load':'Test'}), how = 'outer').plot(
                    y = ['Train', 'Validation', 'Test'], figsize = (15, 10), fontsize = 12)
plt.xlabel('Timestamp', fontsize = 12)
plt.ylabel('Energy Laod', fontsize = 12)

T = 6
HORIZON = 1
scaler = MinMaxScaler()

train_df = load_ts_df.copy().loc[:validation_timestamp].iloc[:-1][['Load']]
train_df['Load'] = scaler.fit_transform(train_df)

train_shifted_df = train_df.copy()
train_shifted_df['Y(t+1)'] = train_shifted_df['Load'].shift(-1, freq = 'H')
for t in range(1, T+1):
    train_shifted_df[str(T-t)] = train_shifted_df['Load'].shift(T-t, freq = 'H')

feature_columns = ['Load(t-5)', 'Load(t-4)', 'Load(t-3)', 'Load(t-2)', 'Load(t-1)', 'Load(t)']
train_shifted_df.columns = ['Load(Observed)', 'Y(t+1)'] + feature_columns
train_shifted_df = train_shifted_df.dropna(how = 'any')

features_train = train_shifted_df[feature_columns].to_numpy()
features_train = np.expand_dims(features_train, axis = 2)
target_train = train_shifted_df['Y(t+1)'].to_numpy()

validation_timestamp_shifted = pd.to_datetime(validation_timestamp) - pd.Timedelta(T-1, 'hours')
validation_df = load_ts_df.copy()[validation_timestamp_shifted:test_timestamp].iloc[:-1][['Load']]
validation_df['Load'] = scaler.transform(validation_df)

validation_shifted_df = validation_df.copy()
validation_shifted_df['Y(t+1)'] = validation_shifted_df['Load'].shift(-1, freq = 'H')
for t in range(1, T+1):
    validation_shifted_df[str(T-t)] = validation_shifted_df['Load'].shift(T-t, freq = 'H')
validation_shifted_df.columns = train_shifted_df.columns
validation_shifted_df = validation_shifted_df.dropna(how = 'any')

features_validation = validation_shifted_df[feature_columns].to_numpy()
features_validation = np.expand_dims(features_validation, axis = 2)
target_validation = validation_shifted_df['Y(t+1)'].to_numpy()

test_timestamp_shifted = pd.to_datetime(test_timestamp) - pd.Timedelta(T-1, 'hours')
test_df = load_ts_df.copy()[test_timestamp_shifted:][['Load']]
test_df['Load'] = scaler.transform(test_df)

test_shifted_df = test_df.copy()
test_shifted_df['Y(t+1)'] = test_shifted_df['Load'].shift(-1, freq = 'H')
for t in range(1, T+1):
    test_shifted_df[str(T-t)] = test_shifted_df['Load'].shift(T-t, freq = 'H')
test_shifted_df.columns = train_shifted_df.columns
test_shifted_df = test_shifted_df.dropna(how = 'any')

features_test = test_shifted_df[feature_columns].to_numpy()
features_test = np.expand_dims(features_test, axis = 2)
target_test = test_shifted_df['Y(t+1)'].to_numpy()

#%% Univariate Modelling Using Gated Recurrent Units (GRUs)
univariate_GRU_model = models.Sequential()
univariate_GRU_model.add(layers.GRU(5, input_shape = (T, 1)))
univariate_GRU_model.add(layers.Dense(HORIZON))

univariate_GRU_model.compile(optimizer = 'RMSprop', loss = 'mse')
print(univariate_GRU_model.summary())

univariate_GRU_model_history = univariate_GRU_model.fit(
    features_train, target_train, batch_size = 16, epochs = 10,
    validation_data = (features_validation, target_validation),
    callbacks = callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5), verbose = 1)

univariate_predictions = univariate_GRU_model.predict(features_test)
evaluation_df = pd.DataFrame(univariate_predictions, columns = ['t+{}'.format(t) for t in range(1, HORIZON+1)])
evaluation_df['timestamp'] = test_shifted_df.index
evaluation_df = pd.melt(evaluation_df, id_vars = 'timestamp', value_name = 'prediction', var_name = 'h')
evaluation_df['actual'] = np.transpose(target_test).ravel()
evaluation_df[['actual', 'prediction']] = scaler.inverse_transform(evaluation_df[['actual', 'prediction']])

def MAPE(actual, predicted):
    return (np.abs(actual - predicted)/actual).mean()

print('Univariate Model: MAPE = {:.4f}%, Accuracy = {:.2f}%'.format(
    MAPE(evaluation_df['actual'], evaluation_df['prediction']) * 100,
    (1 - MAPE(evaluation_df['actual'], evaluation_df['prediction'])) * 100))

evaluation_df.loc[evaluation_df['timestamp'] < '2014-11-08'].plot(
    x = 'timestamp', y = ['actual', 'prediction'], style = ['b', 'r--'], figsize = (15, 10), fontsize = 12)
plt.xlabel('Timestamp', fontsize = 12)
plt.ylabel('Energy Load', fontsize = 12)
plt.legend(['Actual', 'Prediction'])

#%% Multivariate Modelling
multivariate_train_df = energy_ts_df.copy()[:validation_timestamp].iloc[:-1][['Load', 'Temperature']]
multivariate_scaler = MinMaxScaler()
multivariate_train_df[['Load', 'Temperature']] = multivariate_scaler.fit_transform(multivariate_train_df)

tensor = {'Features': (range(-T+1, 1), ['Load', 'Temperature'])}
multivariate_train_input = TimeSeriesTensor(dataset = multivariate_train_df, target = 'Load',
                                            H = HORIZON, tensor_structure = tensor, freq = 'H',
                                            drop_incomplete = True)

multivariate_validation_df = energy_ts_df.copy().loc[validation_timestamp_shifted:test_timestamp].iloc[
    :-1][['Load', 'Temperature']]
multivariate_validation_df[['Load', 'Temperature']] = multivariate_scaler.transform(multivariate_validation_df)
multivariate_validation_input = TimeSeriesTensor(dataset = multivariate_validation_df, target = 'Load',
                                                 H = HORIZON, tensor_structure = tensor)

multivariate_GRU_model = models.Sequential()
multivariate_GRU_model.add(layers.GRU(5, input_shape = (T, 2)))
multivariate_GRU_model.add(layers.Dense(HORIZON))

multivariate_GRU_model.compile(optimizer = 'RMSprop', loss = 'mse')
print(multivariate_GRU_model.summary())

multivariate_GRU_model_history = multivariate_GRU_model.fit(
    multivariate_train_input['Features'], multivariate_train_input['target'], batch_size = 16, epochs = 10,
    validation_data = [multivariate_validation_input['Features'], multivariate_validation_input['target']],
    callbacks = callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5), verbose = 1)

multivariate_test_df = energy_ts_df.copy().loc[test_timestamp_shifted:][['Load', 'Temperature']]
multivariate_test_df[['Load', 'Temperature']] = multivariate_scaler.transform(multivariate_test_df)
multivariate_test_input = TimeSeriesTensor(dataset = multivariate_test_df, target = 'Load', H = HORIZON,
                                           tensor_structure = tensor)

multivariate_predictions = multivariate_GRU_model.predict(multivariate_test_input['Features'])
multivariate_evaluation_df = create_evaluation_df(multivariate_predictions, multivariate_test_input,
                                                  HORIZON, scaler)
print('Multivariate Model: MAPE = {:.4f}%, Accuracy = {:.2f}%'.format(
    MAPE(multivariate_evaluation_df['actual'], multivariate_evaluation_df['prediction']) * 100,
    (1 - MAPE(multivariate_evaluation_df['actual'], multivariate_evaluation_df['prediction'])) * 100))