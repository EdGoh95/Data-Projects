#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 7 - Experimentation in action: Moving from prototype to MVP
"""
import os
os.environ['SPARK_REMOTE'] = "sc://localhost"
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, Trials, tpe
from helper_functions import get_airport_data, apply_index_frequency, generate_splits,\
    HWES, HWES_optimization_function, run_hyperparameter_tuning, get_airport_data_spark,\
        get_all_airports_spark, HWES_optimization_function_spark, run_hyperparameter_tuning_cluster,\
            run_hyperparameter_tuning_udf
from pyspark.sql import types, functions, SparkSession

#%% Introduction To HyperOpt (Tree-Structured Parzen Estimators - TPEs)
def objective_function(x):
    func = np.poly1d([1, -3, -88, 112, -5])
    return func(x) * 0.01

trials = Trials()
trial_estimator = fmin(fn = objective_function, space = hp.uniform('x', -12, 12), algo = tpe.suggest,
                       trials = trials, max_evals = 1000)

rng = np.arange(-11.0, 12.0, 0.01)
values = [objective_function(x) for x in rng]
with plt.style.context(style = 'seaborn'):
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    ax.plot(rng, values)
    ax.set_title('Objective Function')
    ax.scatter(x = trial_estimator['x'], y = trials.average_best_error(), marker = 'o', s = 100)
    bbox_text = 'Minimum value calculated using Hyperopt\nx = {}'.format(trial_estimator['x'])
    configuration = dict(xycoords = 'data', textcoords = 'axes fraction',
                          arrowprops = dict(facecolor = 'aqua', shrink = 0.01, connectionstyle = 'angle3'),
                          bbox = dict(boxstyle = 'round,pad=0.5', fc = 'ivory', ec = 'grey', lw = 0.8),
                          ha = 'left', va = 'center', fontsize = 12)
    ax.annotate(bbox_text, xy = (trial_estimator['x'], trials.average_best_error()),
                xytext = (0.3, 0.8), **configuration)
    bbox_text.get_window_extent(fig.canvas.get_renderer())
    fig.tight_layout()

#%% Apply HyperOpt To Optimize The Hyperparameters For The Time-Series Forecasting
hyperopt_search_space = {'model': {'trend': hp.choice('trend', ['add', 'mul']),
                                   'seasonal': hp.choice('seasonal', ['add', 'mul']),
                                   'seasonal_periods': hp.quniform('seasonal_periods', 12, 120, 12),
                                   'damped_trend': hp.choice('damped_trend', [True, False])},
                         'fit': {'smoothing_level': hp.uniform('smoothing_level', 0.01, 0.99),
                                 'smoothing_seasonal': hp.uniform('smoothing_seasonal', 0.01, 0.99),
                                 'damping_trend': hp.uniform('damping_trend', 0.01, 0.99),
                                 'use_brute': hp.choice('use_brute', [True, False]),
                                 'method': hp.choice('method', ['basinhopping', 'L-BFGS-B']),
                                 'remove_bias': hp.choice('remove_bias', [True, False])}}

parameters = {'optimization_function': HWES_optimization_function, 'tuning_space': hyperopt_search_space,
              'forecast_algo': HWES, 'loss_metric': 'BIC', 'hyperopt_algo': tpe.suggest,
              'iterations': 400, 'time_series_name': 'JFK (HyperOpt)', 'value_name': 'Total Passengers',
              'image_name': 'JFK Total Passengers (Hyperparameter Tuning Using HyperOpt).svg'}

JFK_airport_data = get_airport_data('JFK', '../TCPD/datasets/jfk_passengers/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv')
JFK_airport_data = apply_index_frequency(JFK_airport_data, 'MS')
JFK_training_dataset, JFK_testing_dataset = generate_splits(JFK_airport_data, '2014-12-01')
JFK_hyperopt = run_hyperparameter_tuning(JFK_training_dataset['Total Passengers'],
                                         JFK_testing_dataset['Total Passengers'], parameters)

#%% HyperOpt Using A Spark Cluster
spark_session = SparkSession.builder.remote(
    'sc://6112322794996208.8.gcp.databricks.com:443/;token=dapi0f92a60bfc652437060b4427a5fbecac;x-databricks-cluster-id=1222-094539-jwcmrww4').getOrCreate()

csv_file_location = 'dbfs:/user/hive/warehouse/air-passenger-traffic-per-month-port-authority-of-ny-nj-beginning-1977.csv'
database_name = 'edwingoh95'
table_name = 'all_airports'
full_name = '{}.{}'.format(database_name, table_name)

airport_schema = types.StructType([types.StructField('Airport_Code', types.StringType()),
                                   types.StructField('Year', types.StringType()),
                                   types.StructField('Month', types.StringType()),
                                   types.StructField('Domestic_Passengers', types.IntegerType()),
                                   types.StructField('International_Passengers', types.IntegerType()),
                                   types.StructField('Total_Passengers', types.IntegerType())])
all_airport_traffic = spark_session.read.csv(csv_file_location, header = True, inferSchema = False,
                                             schema = airport_schema).withColumn('day', functions.lit('1'))
all_airport_traffic_formatted = all_airport_traffic.withColumn('Raw Date', functions.concat_ws(
    '-', *[functions.col('Year'), functions.col('Month'), functions.col('Day')])).withColumn(
        'Date', functions.to_date(functions.col('Raw Date'), format = 'yyyy-MMM-d')).drop(
            *['Year', 'Month', 'Day', 'Raw Date'])
all_airport_traffic_formatted.write.format('delta').mode('overwrite').option('mergeSchema', 'true').option(
    'overwriteSchema', 'true').partitionBy('Airport_Code').saveAsTable(full_name)

output_schema = types.StructType([types.StructField('Airport', types.StringType()),
                                   types.StructField('Date', types.DateType()),
                                   types.StructField('Total_Passengers_Forecast', types.IntegerType()),
                                   types.StructField('Is_Future', types.BooleanType())])

@functions.pandas_udf(output_schema, functions.PandasUDFType.GROUPED_MAP)
def airport_forecast(airport):
    airport_data = get_airport_data_spark(full_name)
    airport_name = airport_data['Airport_Code'][0]
    spark_run_parameters = {'optimization_function': HWES_optimization_function_spark,
                            'tuning_space': hyperopt_search_space, 'forecast_algo': HWES,
                            'loss_metric': 'BIC', 'hyperopt_algo': tpe.suggest,
                            'timeout': 1800, 'iterations': 600, 'experiment_name': 'Airport_Forecast',
                            'time_series_name': '{} (HyperOpt)'.format(airport_name),
                            'value_name': 'Total Passengers',
                            'image_name': '{} Total Passengers (Hyperparameter Tuning Using HyperOpt).svg'
                            .format(airport_name)}
    airport_data = apply_index_frequency(airport_data, 'MS').fillna(method = 'ffill').fillna(method = 'bfill')
    airport_training_dataset, airport_testing_dataset = generate_splits(airport_data, '2014-12-01')
    airport_hyperopt = run_hyperparameter_tuning_udf(airport_training_dataset['Total_Passengers'],
                                                     airport_testing_dataset['Total_Passengers'],
                                                     spark_run_parameters)
    return airport_hyperopt
def validate_data_counts(data, split_count):
    return list(data.groupBy(functions.col('Airport_Code')).count().withColumn(
        'check', functions.when(((functions.lit(12)/0.2) < (functions.col('count') * 0.8)), True).otherwise(
            False)).filter(functions.col('check')).select('Airport_Code').toPandas()['Airport_Code'])

airport_traffic_dataset = spark_session.table(full_name)
filtered_dataset = airport_traffic_dataset.where(
    functions.col('Airport_Code').isin(validate_data_counts(airport_traffic_dataset, 12))).repartition('Airport_Code')

# best_hyperparameters = {}
# for airport_name in airports:
#     spark_run_parameters = {'optimization_function': HWES_optimization_function,
#                             'tuning_space': hyperopt_search_space, 'forecast_algo': HWES,
#                             'loss_metric': 'BIC', 'hyperopt_algo': tpe.suggest, 'parallelism': 64,
#                             'timeout': 3600, 'iterations': 1000, 'airport_name': airport_name,
#                             'experiment_name': 'Airport_Forecast',
#                             'time_series_name': '{} (HyperOpt)'.format(airport_name),
#                             'value_name': 'Total Passengers',
#                             'image_name': '{} Total Passengers (Hyperparameter Tuning Using HyperOpt).png'.format(airport_name),
#                             'hyperopt_image_name': '{} Total Passengers (Hyperparameter Tuning Using HyperOpt).png'.format(airport_name)}
#     print('Importing data for {}...'.format(airport_name))
#     airport_data = get_airport_data_spark(spark, airport_name, full_name)
#     airport_data = apply_index_frequency(airport_data, 'MS').fillna(
#         method = 'ffill').fillna(method = 'bfill')
#     training_dataset, test_dataset = generate_splits(airport_data, '2014-12-01')
#     print('Running Hyperopt for {}...'.format(airport_name))
#     airport_hyperopt = run_hyperparameter_tuning_cluster(training_dataset['Total_Passengers'],
#                                                          test_dataset['Total_Passengers'], spark_run_parameters)
#     print('Best hyperparameters for {}: {}'.format(airport_name, airport_hyperopt['best_hyperparameters']))
#     best_hyperparameters[airport_name] = airport_hyperopt['best_hyperparameters']
#
# print(best_hyperparameters)


grouped_apply = filtered_dataset.groupBy('Airport_Code').apply(airport_forecast)
grouped_apply.show(grouped_apply.count(), False)