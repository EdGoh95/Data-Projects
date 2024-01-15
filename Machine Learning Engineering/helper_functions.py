#!/usr/bin/env python3
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import uuid
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
from hyperopt import fmin, Trials, space_eval, STATUS_OK, SparkTrials
from functools import partial

def apply_index_frequency(data, frequency):
    '''
    Setting the time-series frequency as the index of the dataframe
    '''
    return data.asfreq(frequency)

def pull_raw_airport_data(file_location):
    airport_traffic_df = pd.read_csv(file_location)
    airport_traffic_df = airport_traffic_df.copy(deep = False)
    airport_traffic_df['Month'] = pd.to_datetime(airport_traffic_df['Month'], format = '%b').dt.month
    airport_traffic_df.loc[:, 'Day'] = 1
    airport_traffic_df['date'] = pd.to_datetime(airport_traffic_df[['Year', 'Month', 'Day']])
    airport_traffic_df.set_index('date', inplace = True)
    airport_traffic_df.index = pd.DatetimeIndex(airport_traffic_df.index.values,
                                                freq = airport_traffic_df.index.inferred_freq)
    return airport_traffic_df.sort_index()

def get_raw_airport_data_spark(spark, full_name):
    airport_traffic_df = spark.table(full_name).toPandas()
    airport_traffic_df = airport_traffic_df.copy(deep = True)
    airport_traffic_df['Date'] = pd.to_datetime(airport_traffic_df['Date'])
    airport_traffic_df.set_index('Date', inplace = True)
    airport_traffic_df.index = pd.DatetimeIndex(airport_traffic_df.index.values,
                                                freq = airport_traffic_df.index.inferred_freq)
    return airport_traffic_df.sort_index()

def get_airport_data(airport_name, file_location):
    all_airport_traffic_data = pull_raw_airport_data(file_location)
    return all_airport_traffic_data[all_airport_traffic_data['Airport Code'] == airport_name]

def get_airport_data_spark(spark, airport_name, full_name):
    all_airport_traffic_data = get_raw_airport_data_spark(spark, full_name)
    return all_airport_traffic_data[all_airport_traffic_data['Airport_Code'] == airport_name]

def get_all_airports(file_location):
    all_airport_traffic_data = pull_raw_airport_data(file_location)
    unique_airports = all_airport_traffic_data['Airport Code'].unique()
    return np.sort(unique_airports)

def get_all_airports_spark(full_name):
    all_airport_traffic_data = get_raw_airport_data_spark(full_name)
    unique_airports = all_airport_traffic_data['Airport_Code'].unique()
    return np.sort(unique_airports)

def generate_outlier_plots(data_series, series_name, series_column, event_date, event_name, image_name):
    log_series_name = 'Log  {}'.format(series_column)
    monthly_log_series_name = 'DiffLog {} By Month'.format(series_column)
    yearly_log_series_name = 'DiffLog {} By Year'.format(series_column)
    event_boundary_low = datetime.datetime.strptime(event_date, '%Y-%m-%d').replace(day = 1) - relativedelta(months = 2)
    event_boundary_high = datetime.datetime.strptime(event_date, '%Y-%m-%d').replace(day = 1) + relativedelta(months = 2)
    max_scaling = np.round(data_series[series_column].values.max() * 1.1, 0)
    data = data_series.copy(deep = True)

    data[log_series_name] = np.log(data[series_column])
    data[monthly_log_series_name] = data[log_series_name].diff(1)
    data[yearly_log_series_name] = data[log_series_name].diff(12)

    fig, axes = plt.subplots(3, 1, figsize = (15, 8.5), constrained_layout = True)
    axes[0].plot(data[log_series_name], '-', label = series_column)
    axes[0].set(title = '{} {}'.format(series_name, series_column))
    axes[0].axvline(event_boundary_low, 0, max_scaling, color = 'red', linestyle = '--', label = event_name)
    axes[0].axvline(event_boundary_high, 0, max_scaling, color = 'red', linestyle = '--')
    axes[0].legend(loc = 'best')

    axes[1].plot(data[monthly_log_series_name],label = 'Monthly Difference In {}'.format(series_column))
    axes[1].hlines(0, data.index[0], data.index[-1], 'green')
    axes[1].set(title = '{} Monthly Difference of {}'.format(series_name, series_column))
    axes[1].axvline(event_boundary_low, 0, max_scaling, color = 'red', linestyle = '--', label = event_name)
    axes[1].axvline(event_boundary_high, 0, max_scaling, color = 'red', linestyle = '--')
    axes[1].legend(loc = 'best')

    axes[2].plot(data[yearly_log_series_name],label = 'Yearly Difference In {}'.format(series_column))
    axes[2].hlines(0, data.index[0], data.index[-1], 'green')
    axes[2].set(title = '{} Yearly Difference of {}'.format(series_name, series_column))
    axes[2].axvline(event_boundary_low, 0, max_scaling, color = 'red', linestyle = '--', label = event_name)
    axes[2].axvline(event_boundary_high, 0, max_scaling, color = 'red', linestyle = '--')
    axes[2].legend(loc = 'best')

    plt.savefig(image_name, format = 'svg', dpi = 600)
    return fig

def stationary_tests(time_series_df, series_column, time_series_name, period, image_name, lags = 12,
                     cf_alpha = 0.05, style = 'seaborn', plot_size = (15, 30)):
    log_column_name = 'Log {}'.format(series_column)
    diff_log_column_name = 'DiffLog {}'.format(series_column)
    time_series_df[log_column_name] = np.log(time_series_df[series_column])
    time_series_df[diff_log_column_name] = time_series_df[log_column_name].diff()
    trend_decomposition = seasonal_decompose(time_series_df[series_column], period = period)
    start_index = time_series_df.index.values[0]
    end_index = time_series_df.index.values[len(time_series_df) - 1]
    with plt.style.context(style = style):
        fig, axes = plt.subplots(7, 1, figsize = plot_size, constrained_layout = True)
        plt.subplots_adjust(hspace = 0.3)
        axes[0].plot(time_series_df[series_column], '-', label = '{} For {}'.format(series_column, time_series_name))
        axes[0].legend(loc = 'upper left')
        axes[0].set_title('{} {} Prior To Time Series Decomposition'.format(time_series_name, series_column))
        axes[0].set_xlabel(time_series_df.index.name)
        axes[0].set_ylabel(series_column)

        axes[1].plot(time_series_df[diff_log_column_name], 'g-',
                     label = 'Log Difference In {}'.format(series_column))
        axes[1].hlines(0, start_index, end_index, 'red', label = 'Series Centre')
        axes[1].legend(loc = 'lower left')
        axes[1].set_title('{} DiffLog Trend Of Outliers In {} Prior To Time Series Decomposition'
                          .format(time_series_name, series_column))
        axes[1].set_xlabel(time_series_df.index.name)
        axes[1].set_ylabel(series_column)

        fig = plot_acf(time_series_df[series_column], lags = lags, ax = axes[2])
        fig = plot_pacf(time_series_df[series_column], lags = lags, ax = axes[3])
        axes[2].set_xlabel('Lags')
        axes[2].set_ylabel('Correlation')
        axes[3].set_xlabel('Lags')
        axes[3].set_ylabel('Correlation')

        axes[4].plot(trend_decomposition.trend, 'r-', label = 'Trend For {}'.format(time_series_name))
        axes[4].legend(loc = 'upper left')
        axes[4].set_title('Trend Component Of {} After Time Series Decomposition'.format(time_series_name))
        axes[4].set_xlabel(time_series_df.index.name)
        axes[4].set_ylabel(series_column)

        axes[5].plot(trend_decomposition.seasonal, 'r-', label = 'Seasonal Pattern Of {} For {}'
                     .format(series_column, time_series_name))
        axes[5].legend(loc = 'upper left')
        axes[5].set_title('Seasonal Component Of {} After Time Series Decomposition'.format(time_series_name))
        axes[5].set_xlabel(time_series_df.index.name)
        axes[5].set_ylabel(series_column)

        axes[6].plot(trend_decomposition.resid, 'r.', label = 'Residual Pattern Of {} For {}'
                     .format(series_column, time_series_name))
        axes[6].hlines(0, start_index, end_index, 'black', label = 'Series Centre')
        axes[6].legend(loc = 'upper left')
        axes[6].set_title('Residuals Of {} For {} After Time Series Decomposition'
                          .format(series_column, time_series_name))
        axes[6].set_xlabel(time_series_df.index.name)
        axes[6].set_ylabel(series_column)

        plt.tight_layout()
        plt.savefig(image_name, format = 'svg', dpi = 600)
    return fig

def exponential_smoothing(raw_time_series, alpha = 0.05):
    output = [raw_time_series[0]]
    for i in range(1, len(raw_time_series)):
        output.append((raw_time_series[i] * alpha) + ((1 - alpha) * output[i-1]))
    return output

def calculate_MAE(raw_time_series, smoothed_time_series, window, scale):
    results = {}
    MAE_value = mean_absolute_error(raw_time_series[window:], smoothed_time_series[window:])
    results['MAE'] = MAE_value
    deviation = np.std(raw_time_series[window:] - smoothed_time_series[window:])
    results['stddev'] = deviation
    yhat = MAE_value + (scale * deviation)
    results['yhat_lower'] = smoothed_time_series - yhat
    results['yhat_upper'] = smoothed_time_series + yhat
    return results

def smoothed_time_series_plots(time_series, time_series_name, image_name, smoothing_window,
                               exponential_alpha = 0.05, yhat_scale = 1.96, style = 'seaborn',
                               plot_size = (15, 22.5)):
    reference_collection = {}
    ts = pd.Series(time_series)
    with plt.style.context(style = style):
        fig, axes = plt.subplots(3, 1, figsize = plot_size)
        plt.subplots_adjust(hspace = 0.3)
        moving_average = ts.rolling(window = smoothing_window).mean()
        exponential_smoothed = exponential_smoothing(ts, exponential_alpha)
        result = calculate_MAE(time_series, moving_average, smoothing_window, yhat_scale)
        exponential_smoothed_result = calculate_MAE(time_series, exponential_smoothed, smoothing_window, yhat_scale)
        exponential_smoothed_df = pd.Series(exponential_smoothed, index = time_series.index)
        exponential_smoothed_lower_df = pd.Series(exponential_smoothed_result['yhat_lower'],
                                                  index = time_series.index)
        exponential_smoothed_upper_df = pd.Series(exponential_smoothed_result['yhat_upper'],
                                                  index = time_series.index)

        axes[0].plot(ts, '-', label = 'Raw Trend For {}'.format(time_series_name))
        axes[0].set_title('{} {} Prior To Smoothing'.format(time_series_name, time_series.name))
        axes[0].legend(loc = 'upper left')

        axes[1].plot(ts, '-', label = 'Raw Trend For {}'.format(time_series_name))
        axes[1].plot(moving_average, 'g-', label = 'Moving Average With Window Size = {}'.format(smoothing_window))
        axes[1].plot(result['yhat_upper'], 'r--', label = r'$\^y$ bounds')
        axes[1].plot(result['yhat_lower'], 'r--')
        axes[1].set_title('Moving Average Of {} For {} (Window Size = {}, MAE = {:.1f})'.format(
            time_series.name, time_series_name, smoothing_window, result['MAE']))
        axes[1].legend(loc = 'upper left')

        axes[2].plot(ts, '-', label = 'Raw Trend For {}'.format(time_series_name))
        axes[2].plot(exponential_smoothed_df, 'g-',
                     label = r'Exponential Smoothing With $\alpha$ = {}'.format(exponential_alpha))
        axes[2].plot(exponential_smoothed_upper_df, 'r--', label = r'$\^y$ bounds')
        axes[2].plot(exponential_smoothed_lower_df, 'r--')
        axes[2].set_title(r'Exponential Smoothing Of {} For {} ($\alpha$ = {}, MAE = {:.1f})'.format(
            time_series.name, time_series_name, exponential_alpha, exponential_smoothed_result['MAE']))
        axes[2].legend(loc = 'upper left')

        plt.tight_layout()
        plt.savefig(image_name, format = 'svg', dpi = 600)
        reference_collection['Plots'] = fig
        reference_collection['Moving Average'] = moving_average
        reference_collection['Exponential Smoothing'] = exponential_smoothed
        return reference_collection

def Holt_Winters_Exponential_Smoothing(train, test, seasonal, trend, periods, damping, smoothing_slope,
                                       damping_slope):
    output = {}
    exponential_smoothing_model = ExponentialSmoothing(train, trend = trend, seasonal = seasonal,
                                                       seasonal_periods = periods, damped = damping)
    model_fit = exponential_smoothing_model.fit(
        smoothing_level = 0.9, smoothing_seasonal = 0.2, smoothing_trend = smoothing_slope,
        damping_trend = damping_slope, use_brute = True, method = 'basinhopping', remove_bias = True)
    model_forecast = model_fit.predict(train.index[-1], test.index[-1])
    output['Model'] = model_fit
    output['Forecast'] = model_forecast[1:]
    return output

# Scoring & Evaluation Metrics
def MAPE(y_actual, y_predicted):
    return np.mean(np.abs((y_actual - y_predicted)/y_actual)) * 100

def AIC(n, MSE, param_count):
    return (n * np.log(MSE)) + (2 * param_count)

def BIC(n, MSE, param_count):
    return (n * np.log(MSE)) + (param_count * np.log(n))

def calculate_errors(y_actual, y_predicted, param_count):
    error_scores = {}
    error_scores['MAE'] = mean_absolute_error(y_actual, y_predicted)
    error_scores['MAPE'] = MAPE(y_actual, y_predicted)
    error_scores['MSE'] = mean_squared_error(y_actual, y_predicted, squared = True)
    error_scores['RMSE'] = mean_squared_error(y_actual, y_predicted, squared = False)
    error_scores['Explained Variance'] = explained_variance_score(y_actual, y_predicted)
    error_scores['R2'] = r2_score(y_actual, y_predicted)
    error_scores['AIC'] = AIC(len(y_predicted), error_scores['MSE'], param_count)
    error_scores['BIC'] = BIC(len(y_predicted), error_scores['MSE'], param_count)
    return error_scores

def plot_predictions(y_actual, y_predicted, param_count, time_series_name, value_name, image_name,
                     style = 'seaborn', plot_size = (12, 9)):
    validation_output = {}
    error_values = calculate_errors(y_actual, y_predicted, param_count)
    validation_output['Errors'] = error_values
    text = '\n'.join(('MAE = {:.3f}'.format(error_values['MAE']),
                      'MAPE = {:.3f}'.format(error_values['MAPE']),
                      'MSE = {:.3f}'.format(error_values['MSE']),
                      'RMSE = {:.3f}'.format(error_values['RMSE']),
                      'Explained Variance = {:.3f}'.format(error_values['Explained Variance']),
                      'R Squared = {:.3f}'.format(error_values['R2']),
                      'AIC = {:.3f}'.format(error_values['AIC']),
                      'BIC = {:.3f}'.format(error_values['BIC'])))

    with plt.style.context(style = style):
        fig, axes = plt.subplots(1, 1, figsize = plot_size)
        axes.plot(y_actual, 'b-', label = 'Actual {} For {}'.format(value_name, time_series_name))
        axes.plot(y_predicted, 'r-', label = 'Forecast Of {} For {}'.format(value_name, time_series_name))
        axes.legend(loc = 'upper left')
        axes.set_title('Actual And Predicted {} For {}'.format(value_name, time_series_name))
        axes.set_xlabel(y_actual.index.name)
        axes.set_ylabel(value_name)
        axes.text(0.05, 0.9, text, transform = axes.transAxes, fontsize = 12, verticalalignment = 'top',
                  bbox = dict(boxstyle = 'round', facecolor = 'oldlace', alpha = 0.5))
        validation_output['Plot'] = fig
        plt.tight_layout()
        plt.savefig(image_name, format = 'png', dpi = 600)
    return validation_output

def split_correctness(data, train, test):
    '''
    Validation assertion function designed to ensure that the splits being conducted through the
    custom function are not dropping any rows of data between the training and test datasets

    '''
    assert data.size == train.size + test.size, ('The combined size of training ({}) and test ({})' + \
        ' datasets did not match to the size of the original dataset({})').format(train.size, test.size, data.size)

def generate_splits(data, date):
    parsed_date = parse(date, fuzzy = True)
    nearest_date = data[:parsed_date].iloc(0)[-1].name
    train = data[:nearest_date]
    test = data[nearest_date:][1:]
    split_correctness(data, train, test)
    return train, test

# Hyperparameter Tuning
def extract_param_count_HWES(config):
    return len(config['model'].keys()) + len(config['fit'].keys())

def HWES(hp_values, train, test):
    output = {}
    model = ExponentialSmoothing(train, trend = hp_values['model']['trend'],
                                 seasonal = hp_values['model']['seasonal'],
                                 seasonal_periods = hp_values['model']['seasonal_periods'],
                                 damped_trend = hp_values['model']['damped_trend'])
    model_fit = model.fit(smoothing_level = hp_values['fit']['smoothing_level'],
                          smoothing_seasonal = hp_values['fit']['smoothing_seasonal'],
                          damping_trend = hp_values['fit']['damping_trend'],
                          use_brute = hp_values['fit']['use_brute'], method = hp_values['fit']['method'],
                          remove_bias = hp_values['fit']['remove_bias'])
    forecast = model_fit.predict(train.index[-1], test.index[-1])
    output['model'] = model_fit
    output['forecast'] = forecast[1:]
    return output

def HWES_optimization_function(hp_values, train, test, loss_metric):
    model_results = HWES(hp_values, train, test)
    errors = calculate_errors(test, model_results['forecast'], extract_param_count_HWES(hp_values))
    mlflow.log_params(hp_values)
    mlflow.log_metrics(errors)
    return {'loss': errors[loss_metric], 'status': STATUS_OK}

def HWES_optimization_function_spark(hp_values, train, test, loss_metric, airport, experiment_name,
                                     trial):
    model_results = HWES(hp_values, train, test)
    errors = calculate_errors(test, model_results['forecast'], extract_param_count_HWES(hp_values))
    with mlflow.start_run(run_name = '{}_{}_{}_{}'.format(
            airport, experiment_name, str(uuid.uuid4())[:8], trial.results)):
        mlflow.set_tag('airport', airport)
        mlflow.set_tag('parent_run', experiment_name)
        mlflow.log_param('id', mlflow.active_run().info.run_id)
        mlflow.log_params(hp_values)
        mlflow.log_metrics(errors)
    return {'loss': errors[loss_metric], 'status': STATUS_OK}

def run_hyperparameter_tuning(train, test, params):
    param_count = extract_param_count_HWES(params['tuning_space'])
    output = {}
    trial_run = Trials()
    hyperparameter_tuning = fmin(fn = partial(params['optimization_function'], train = train, test = test,
                                              loss_metric = params['loss_metric']),
                                 space = params['tuning_space'], algo = params['hyperopt_algo'],
                                 max_evals = params['iterations'], trials = trial_run)
    best_run = space_eval(params['tuning_space'], hyperparameter_tuning)
    generated_model = params['forecast_algo'](best_run, train, test)
    output['best_hyperparameters'] = best_run
    output['best_model'] = generated_model['model']
    output['forecast'] = generated_model['forecast']
    output['plot'] = plot_predictions(test, generated_model['forecast'], param_count,
                                      params['time_series_name'], params['value_name'],
                                      params['image_name'])
    return output

def run_hyperparameter_tuning_cluster(train, test, params):
    param_count = extract_param_count_HWES(params['tuning_space'])
    output = {}
    trial_run = SparkTrials(parallelism = params['parallelism'], timeout = params['timeout'])
    with mlflow.start_run(run_name = 'PARENT_RUN_{}'.format(params['airport_name']), nested = True):
        mlflow.set_tag('airport', params['airport_name'])
        hyperparameter_tuning = fmin(fn = partial(
            params['optimization_function'], train = train, test = test,
            loss_metric = params['loss_metric']), space = params['tuning_space'],
            algo = params['hyperopt_algo'], max_evals = params['iterations'], trials = trial_run)
        best_run = space_eval(params['tuning_space'], hyperparameter_tuning)
        generated_model = params['forecast_algo'](best_run, train, test)
        output['best_hyperparameters'] = best_run
        output['best_model'] = generated_model['model']
        output['forecast'] = generated_model['forecast']
        output['plot'] = plot_predictions(test, generated_model['forecast'], param_count,
                                          params['time_series_name'], params['value_name'],
                                          params['image_name'])
        mlflow.log_artifact(params['image_name'])
        mlflow.log_artifact(params['hyperopt_image_name'])
    return output

def run_hyperparameter_tuning_udf(train, test, params):
    param_count = extract_param_count_HWES(params['tuning_space'])
    output = {}
    trial_run = Trials()
    hyperparameter_tuning = fmin(fn = partial(
        params['optimization_function'], train = train, test = test, loss_metric = params['loss_metric'],  airport = params['airport_name'], 
        experiment_name = params['experiment_name'], trial = trial_run), space = params['tuning_space'], algo = params['hyperopt_algo'], 
                                 max_evals = params['iterations'], trials = trial_run)
    best_run = space_eval(params['tuning_space'], hyperparameter_tuning)
    generated_model = params['forecast_algo'](best_run, train, test)
    output['best_hyperparameters'] = best_run
    output['best_model'] = generated_model['model']
    output['forecast'] = generated_model['forecast']
    output['plot'] = plot_predictions(test, generated_model['forecast'], param_count, 
                                      params['time_series_name'], params['value_name'],
                                      params['image_name'])
    return output
