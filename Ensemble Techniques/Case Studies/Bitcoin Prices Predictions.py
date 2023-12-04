#!/usr/bin/env python3
"""
Hands-On Ensemble Learning with Python (Packt Publishing) Chapter 10:
Predicting Bitcoin Prices
"""
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import matplotlib.pyplot as plt
import time
import sys
sys.path.append(
    '../../../../Hands-On Ensemble Learning with Python (Master)/Chapter10/')
sys.path.append('../')
from termcolor import colored
colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'grey', 'white']
from statsmodels.graphics.tsaplots import plot_acf
from simulator import simulate
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from voting_regressor import VotingRegressor
from stacking_regressor import StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

start = time.time()
#%% Load The Dataset
BTC_df = pd.read_csv('Data/Bitcoin Prices In USD (300619).csv').dropna()
BTC_df['Date'] = pd.to_datetime(BTC_df['Date'])
BTC_df.set_index('Date', drop = True, inplace = True)

#%% Data Exploration and Analysis
BTC_df.plot(y = 'Close', title = 'Bitcoin (BTC) Price in USD', xlabel = 'Date', 
            ylabel = 'USD per BTC')
plot_acf(BTC_df['Close'], lags = 30)
plt.xlabel('Lags')
plt.ylabel('Correlation')

#### Transforming The Data To Achieve A Stationary Time Series 
# Reduce Auto-Correlation
percentage_change = BTC_df['Close'].diff()/BTC_df['Close']
plt.figure()
percentage_change.plot(title = 'Transformed Data', xlabel = 'Date', 
                       ylabel = 'Change/%')

fig, (ax1, ax2) = plt.subplots(2, 1)
rolling_std = percentage_change.rolling(30).std()
rolling_std.plot(title = 'Rolling Standard Deviation (Transformed Data)', 
                 ax = ax1)
plot_acf(percentage_change.dropna(), lags = 30, ax = ax2)

#%% Establishing A Baseline Using A Linear Regression Model
LR = LinearRegression()
percentage_change = percentage_change.values[1:]

def create_features(lags = 1):
    change = np.zeros((len(percentage_change), lags))
    for lag in range(1, lags + 1):
        change[lag:, lag - 1] = percentage_change[:-lag]
    return change

# Ensure reproducibility
features = np.round(create_features(lags = 20)*100, 8)
target = np.round(percentage_change*100, 8)

train_window = 150
predictions_baseline = np.zeros(len(percentage_change) - train_window)
for i in range(len(percentage_change) - train_window - 1):
    train_features = features[i:i + train_window, :]
    train_target = target[i:i + train_window]
    LR.fit(train_features, train_target)
    predictions_baseline[i] = LR.predict(
        features[i + train_window + 1, :].reshape(1, -1))
    
print('\u2500'*35 + ' Establishing A Baseline Model ' + '\u2500'*35)
print('Percentage Returns MSE: {:.3f}'.format(mean_squared_error(target[train_window:], 
                                                         predictions_baseline)))
simulate(BTC_df, predictions_baseline)

#%% Ensemble Methods 
#### Voting
base_learners = [('Support Vector Regressor', SVR()), 
                 ('Linear Regression', LR), 
                 ('k-Nearest Neighbours', KNeighborsRegressor())]
voting_ensemble = VotingRegressor(base_learners)
predictions_voting = np.zeros(len(percentage_change) - train_window)
print('\u2500'*36 + ' Different Ensemble Methods ' + '\u2500'*36)
print(colored('Voting', 'blue', attrs = ['bold']))

def create_features_modified(lags = 1):
    change = np.zeros((len(percentage_change), lags))
    moving_average = np.zeros((len(percentage_change), lags))
    
    moving_average_change = (
        BTC_df['Close'].diff()/BTC_df['Close']).rolling(15).mean().fillna(0).values[1:]
    for lag in range(1, lags + 1):
        change[lag:, lag - 1] = percentage_change[:-lag]
        moving_average[lag:, lag - 1] = moving_average_change[:-lag]
    return np.concatenate((change, moving_average), axis = 1)

voting_features = np.round(create_features_modified(lags = 20)*100, 8)
voting_target = np.round(percentage_change*100, 8)

for j in range(len(percentage_change) - train_window - 1):
    voting_train_features = voting_features[j:j + train_window, :]
    voting_train_target = voting_target[j:j + train_window]
    voting_ensemble.fit(voting_train_features, voting_train_target)
    predictions_voting[j] = voting_ensemble.predict(
        voting_features[j + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(voting_target[train_window:], predictions_voting)))
simulate(BTC_df, predictions_voting)

base_learners_modified = base_learners.copy()
base_learners_modified.remove(base_learners[1])
voting_ensemble_modified = VotingRegressor(base_learners_modified)
predictions_voting_modified = np.zeros(len(percentage_change) - train_window)

for k in range(len(percentage_change) - train_window - 1):
    voting_train_features = voting_features[k:k + train_window, :]
    voting_train_target = voting_target[k:k + train_window]
    voting_ensemble_modified.fit(voting_train_features, voting_train_target)
    predictions_voting_modified[k] = voting_ensemble_modified.predict(
        voting_features[k + train_window + 1, :].reshape(1, -1))
print(colored('\nRemoving Linear Regression From The Set of Base Learners:', 
              'grey', attrs = ['bold']))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(voting_target[train_window:], predictions_voting_modified)))
simulate(BTC_df, predictions_voting_modified)

#### Stacking
stacking_ensemble = StackingRegressor([
    [base_learners[l][1] for l in range(len(base_learners))], [LR]])
predictions_stacking = np.zeros(len(percentage_change) - train_window)
print(colored('\nStacking', 'green', attrs = ['bold']))

for m in range(len(percentage_change) - train_window - 1):
    train_features = features[m:m + train_window, :]
    train_target = target[m:m + train_window]
    stacking_ensemble.fit(train_features, train_target)
    predictions_stacking[m] = stacking_ensemble.predict(
        features[m + train_window + 1, :].reshape(1, -1))[-1]
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(target[train_window:], predictions_stacking)))
simulate(BTC_df, predictions_stacking)

stacking_ensemble_modified = StackingRegressor([
    [base_learners_modified[n][1] for n in range(len(base_learners_modified))], 
    [LR]])
predictions_stacking_modified = np.zeros(len(percentage_change) - train_window)
print(colored('\nRemoving Linear Regression From The Set of Base Learners:', 
              'grey', attrs = ['bold']))

for p in range(len(percentage_change) - train_window - 1):
    train_features = features[p:p + train_window, :]
    train_target = target[p:p + train_window]
    stacking_ensemble_modified.fit(train_features, train_target)
    predictions_stacking_modified[p] = stacking_ensemble_modified.predict(
        features[p + train_window + 1, :].reshape(1, -1))[-1]
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(target[train_window:], predictions_stacking_modified)))
simulate(BTC_df, predictions_stacking_modified)

#### Bagging
np.random.seed(123456)
bagging_ensemble = BaggingRegressor()
predictions_bagging = np.zeros(len(percentage_change) - train_window)
print(colored('\nBagging', 'yellow', attrs = ['bold']))

for a in range(len(percentage_change) - train_window - 1):
    train_features = features[a:a + train_window, :]
    train_target = target[a:a + train_window]
    bagging_ensemble.fit(train_features, train_target)
    predictions_bagging[a] = bagging_ensemble.predict(
        features[a + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(target[train_window:], predictions_bagging)))
simulate(BTC_df, predictions_bagging)

np.random.seed(123456)
bagging_ensemble_DT1 = BaggingRegressor(base_estimator = 
                                        DecisionTreeRegressor(max_depth = 1))
predictions_bagging_DT1 = np.zeros(len(percentage_change) - train_window)
print(colored('\nSetting Max Depth of Each Decision Tree = 1:', 'grey', 
              attrs = ['bold']))

for b in range(len(percentage_change) - train_window - 1):
    train_features = features[b:b + train_window, :]
    train_target = target[b:b + train_window]
    bagging_ensemble_DT1.fit(train_features, train_target)
    predictions_bagging_DT1[b] = bagging_ensemble_DT1.predict(
        features[b + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(target[train_window:], predictions_bagging_DT1)))
simulate(BTC_df, predictions_bagging_DT1)

np.random.seed(123456)
bagging_ensemble_DT3 = BaggingRegressor(base_estimator = 
                                        DecisionTreeRegressor(max_depth = 3))
predictions_bagging_DT3 = np.zeros(len(percentage_change) - train_window)
print(colored('\nSetting Max Depth of Each Decision Tree = 3:', 'grey', 
              attrs = ['bold']))

for c in range(len(percentage_change) - train_window - 1):
    train_features = features[c:c + train_window, :]
    train_target = target[c:c + train_window]
    bagging_ensemble_DT3.fit(train_features, train_target)
    predictions_bagging_DT3[c] = bagging_ensemble_DT3.predict(
        features[c + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(target[train_window:], predictions_bagging_DT3)))
simulate(BTC_df, predictions_bagging_DT3)

### Boosting (XGBoost)
boosting_ensemble = XGBRegressor(n_jobs = -1)
predictions_boosting = np.zeros(len(percentage_change) - train_window)
print(colored('\nBoosting', 'magenta', attrs = ['bold']))

for d in range(len(percentage_change) - train_window - 1):
    train_features = features[d:d + train_window, :]
    train_target = target[d:d + train_window]
    boosting_ensemble.fit(train_features, train_target)
    predictions_boosting[d] = boosting_ensemble.predict(
        features[d + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(target[train_window:], predictions_boosting)))
simulate(BTC_df, predictions_boosting)

boosting_ensemble_modified = XGBRegressor(max_depth = 2, n_estimators = 10,
                                          reg_alpha = 0.6, n_jobs = -1)
boosting_features = np.round(create_features_modified(lags = 21)*100, 8)
boosting_target = np.round(percentage_change*100, 8)
predictions_boosting_modified = np.zeros(len(percentage_change) - train_window)
print(colored('\nModifications To Improve Performance', 'grey', attrs = ['bold']))

for e in range(len(percentage_change) - train_window - 1):
    boosting_train_features = boosting_features[e:e + train_window, :]
    boosting_train_target = boosting_target[e:e + train_window]
    boosting_ensemble_modified.fit(boosting_train_features, boosting_train_target)
    predictions_boosting_modified[e] = boosting_ensemble_modified.predict(
        boosting_features[e + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(mean_squared_error(
    boosting_target[train_window:], predictions_boosting_modified)))
simulate(BTC_df, predictions_boosting_modified)

#### Random Forests
np.random.seed(123456)
randforest_ensemble = RandomForestRegressor(n_jobs = -1)
randforest_predictions = np.zeros(len(percentage_change) - train_window)
print(colored('\nRandomForest', 'cyan', attrs = ['bold']))

for f in range(len(percentage_change) - train_window - 1):
    train_features = features[f:f + train_window, :]
    train_target = target[f:f + train_window]
    randforest_ensemble.fit(train_features, train_target)
    randforest_predictions[f] = randforest_ensemble.predict(
        features[f + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(target[train_window:], randforest_predictions)))
simulate(BTC_df, randforest_predictions)

np.random.seed(123456)
randforest_ensemble_modified = RandomForestRegressor(
    max_depth = 2, n_estimators = 51, n_jobs = -1)
randforest_predictions_modified = np.zeros(len(percentage_change) - train_window)
randforest_features = np.round(create_features_modified(lags = 20)*100, 8)
randforest_target = np.round(percentage_change*100, 8)
print(colored('\nModifications To Improve Performance', 'grey', attrs = ['bold']))

for g in range(len(percentage_change) - train_window - 1):
    randforest_train_features = randforest_features[g:g + train_window, :]
    randforest_train_target = randforest_target[g:g + train_window]
    randforest_ensemble_modified.fit(randforest_train_features, randforest_train_target)
    randforest_predictions_modified[g] = randforest_ensemble_modified.predict(
        randforest_features[g + train_window + 1, :].reshape(1, -1))
print('Percentage Returns MSE: {:.3f}'.format(
    mean_squared_error(randforest_target[train_window:], randforest_predictions_modified)))
simulate(BTC_df, randforest_predictions_modified)

stop = time.time()
print('\u2550'*100)
duration = stop - start
minutes = divmod(duration, 60)[0]
seconds = divmod(duration, 60)[1]
print(colored('Execution Duration: {:.2f}s ({:.1f}mins, {:.2f}s)'.format(
    duration, minutes, seconds), 'red'))