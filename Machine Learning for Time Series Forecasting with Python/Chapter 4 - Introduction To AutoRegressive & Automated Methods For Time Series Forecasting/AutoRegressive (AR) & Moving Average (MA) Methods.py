#!/usr/bin/env python3
"""
Machine Learning For Time Series Forecasting With Python
Chapter 4: Introduction To AutoRegressive & Automated Methods For Time Series Forecasting
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.mpl.rc('figure', figsize = (16, 10))
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import ar_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

energy_ts_df = pd.read_csv('../Data/Energy.csv', parse_dates = ['Timestamp'])
energy_ts_df.index = pd.date_range(
    start = min(energy_ts_df['Timestamp']), end = max(energy_ts_df['Timestamp']), freq = 'H')
energy_ts_df = energy_ts_df.drop('Timestamp', axis =  1)

load_ts_df = energy_ts_df[['Load']]

#%% AutoRegression (AR)
plt.figure()
pd.plotting.lag_plot(load_ts_df)

plt.figure()
pd.plotting.autocorrelation_plot(load_ts_df)

load_ts_Aug_2014 = load_ts_df.loc['2014-08-01':'2014-08-07']
plt.figure()
pd.plotting.autocorrelation_plot(load_ts_Aug_2014)

# AutoCorrelation Function (ACF) from the statsmodels library
plot_acf(load_ts_df)
plot_acf(load_ts_Aug_2014)

# Partial AutoCorrelation Function (PACF)
plot_pacf(load_ts_df, lags = 20)
plot_pacf(load_ts_Aug_2014, lags = 30)

# Applying statsmodels' AutoRegressive AR-X(p) model
load_ts_model = ar_model.AutoReg(load_ts_df, lags = 1).fit(cov_type = 'HC0')
print(load_ts_model.summary())

pd.plotting.register_matplotlib_converters()
fig = load_ts_model.plot_predict(720, 840)

diagnostics_plot = load_ts_model.plot_diagnostics(lags = 20)

load_train = load_ts_df.loc['2014-11-01 00:00:00':'2014-12-30 00:00:00'].iloc[:-1]
load_test = load_ts_df.loc['2014-12-30 00:00:00':]

scaler = MinMaxScaler()
load_train['Load'] = scaler.fit_transform(load_train)
load_test['Load'] = scaler.transform(load_test)

HORIZON = 3
print('Forecasting Horizon: {} Hours'.format(HORIZON))
test_shifted = load_test.copy()
for t in range(1, HORIZON):
    test_shifted['Load_' + str(t)] = test_shifted['Load'].shift(-t, freq = 'H')

test_shifted = test_shifted.dropna(how = 'any')

history = [x for x in load_train['Load']]
history = history[-720:]
predictions = []
for t in range(test_shifted.shape[0]):
    load_model = ar_model.AutoReg(history, lags = 1).fit(cov_type = 'HC0')
    yhat = load_model.forecast(steps = HORIZON)
    predictions.append(yhat)
    observed = list(test_shifted.iloc[t])
    history.append(observed[0])
    history.pop(0)
    print('Forecast {} ({}):'.format(t+1, test_shifted.index[t]))
    print('Actual =', observed)
    print('Predicted =', yhat)

#%% AutoRegressive Integrated Moving Average (ARIMA) - Seasonal ARIMA with exogenous factors (SARIMAX)
load_seasonal_train = load_ts_df.loc['2014-09-01 00:00:00':'2014-10-31 23:00:00']
load_seasonal_test = load_ts_df.loc['2014-11-01 00:00:00':]

load_seasonal_train['Load'] = scaler.fit_transform(load_seasonal_train)
load_seasonal_test['Load'] = scaler.transform(load_seasonal_test)

print('\nForecasting Horizon: {} Hours'.format(HORIZON))

load_sarimax_model_test = SARIMAX(endog = load_seasonal_train, order = (4, 1, 0),
                                  seasonal_order = (1, 1, 0, 24)).fit(disp = False)
print(load_sarimax_model_test.summary())

seasonal_test_shifted = load_seasonal_test.copy()
for t in range(1, HORIZON):
    seasonal_test_shifted['Load_' + str(t)] = seasonal_test_shifted['Load'].shift(-t, freq = 'H')

seasonal_test_shifted = seasonal_test_shifted.dropna(how = 'any')

seasonal_history = [x for x in load_seasonal_train['Load']]
seasonal_history = seasonal_history[-720:]
sarimax_predictions = []
for t in range(seasonal_test_shifted.shape[0]):
    load_sarimax_model = SARIMAX(endog = seasonal_history, order = (2, 1, 0),
                                 seasonal_order = (1, 1, 0, 24)).fit(disp = False)
    sarimax_yhat = load_sarimax_model.forecast(steps = HORIZON)
    sarimax_predictions.append(sarimax_yhat)
    observed_seasonal = list(seasonal_test_shifted.iloc[t])
    seasonal_history.append(observed_seasonal[0])
    seasonal_history.pop(0)
    print('Forecast {} ({}):'.format(t+1, seasonal_test_shifted.index[t]))
    print('Actual =', observed_seasonal)
    print('Predicted =', sarimax_yhat)

prediction_df = pd.DataFrame(sarimax_predictions, columns = ['t+{}'.format(t) for t in range(1, HORIZON+1)])
prediction_df['timestamp'] = load_seasonal_test.index[0:len(load_seasonal_test.index)-HORIZON+1]
prediction_df = pd.melt(prediction_df, id_vars = 'timestamp', value_name = 'prediction', var_name = 'h')
prediction_df['actual'] = np.array(np.transpose(seasonal_test_shifted)).ravel()
prediction_df[['actual', 'prediction']] = scaler.inverse_transform(prediction_df[['actual', 'prediction']])

# Compute the mean absolute percentage error (MAPE)
def MAPE(actual, predicted):
    return (np.abs(actual - predicted)/actual).mean()

if HORIZON > 1:
    prediction_df['APE'] = np.abs(prediction_df['actual'] - prediction_df['prediction'])/prediction_df['actual']
    print(prediction_df.groupby('h')['APE'].mean())

one_step_forecast_mape = MAPE(prediction_df[prediction_df['h'] == 't+1']['actual'],
                              prediction_df[prediction_df['h'] == 't+1']['prediction'])
print('One-step forecast MAPE: {:.4f}%'.format(one_step_forecast_mape*100))
print('Multi-step forecast MAPE: {:.4f}%'.format(
    MAPE(prediction_df['actual'], prediction_df['prediction'])*100))