#!/usr/bin/env python3
"""
Machine Learning Engineering In Action (Manning Publication)
Chapter 16 - Production infrastructure
"""
import pandas as pd
import mlflow
from datetime import datetime
from dataclasses import dataclass
from typing import List
from pyspark.sql import functions, SparkSession
from mlflow.tracking.client import MlflowClient
from databricks import feature_store

spark_session = SparkSession.builder.remote(
    'sc://6112322794996208.8.gcp.databricks.com:443/;token=dapi0f92a60bfc652437060b4427a5fbecac;x-databricks-cluster-id=1226-040051-f3rwiwu').getOrCreate()

#%% Artifact Management Using MLflow's Model Registry
@dataclass
class Registry:
    model_name: str
    production_version: int
    updated: bool
    training_time: str()

class RegistryStructure:
    def __init__(self, data):
        self.data = data

    def generate_row(self):
        spark_df = spark_session.createDataFrame(pd.DataFrame([vars(self.data)]))
        return spark_df.withColumn(
            'training_time', functions.to_timestamp(functions.col('training_time'))).withColumn(
                'production_version', functions.col('production_version').cast('long'))

class RegistryLogging:
    def __init__(self, database, table, model_name, production_version, updated):
        self.database = database
        self.table = table
        self.entry_data = Registry(model_name, production_version, updated, self._get_time())

    @classmethod
    def _get_time(self):
        return datetime.today().strfttime('%d/%m/%Y %H:%M:%S')

    def _check_exists(self):
        return spark_session._jsparkSession.catalog().tableExists(self.database, self.table)

    def write_entry(self):
        log_row = RegistryStructure(self.entry_data).generate_row()
        log_row.write.format('delta').mode('append').save(self.delta_location)
        if not self._check_exists():
            spark_session.sql("CREATE TABLE IF NOT EXISTS {}.{};".format(self.database, self.table))

class ModelRegistration:
    def __init__(self, experiment_name, experiment_title, model_name, metric, direction):
        self.experiment_name = experiment_name
        self.experiment_title = experiment_title
        self.model_name = model_name
        self.metric = metric
        self.direction = direction
        self.client = MlflowClient()
        self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    def _get_best_run_info(self, key):
        run_data = mlflow.search_runs(self.experiment_id,
                                      order_by = ['metrics.{} {}'.format(self.metric, self.direction)])
        return run_data.head(1)[key].values[0]

    def _get_registered_status(self):
        return self.client.get_registered_model(name = self.experiment_title)

    def _get_current_production(self):
        return [x.run_id for x in self._get_registered_status().latest_versions
                if x.current_stage == 'Production'][0]

    def _get_production_version(self):
        return int([x.version for x in self.get_registered_status().latest_versions
                    if x.current_stage == 'Production'][0])

    def _get_metric(self, run_id):
        return mlflow.get_run(run_id).data.metrics.get(self.metric)

    def _find_best_model(self):
        try:
            current_production_id = self._get_current_production()
            production_metric = self._get_metric(current_production_id)
        except mlflow.exceptions.RestException:
            current_production_id = -1
            production_metric = 1e9
        best_id = self._get_best_run_info('run_id')
        best_metric = self.get_metric(best_id)
        if self.direction == 'ASC':
            if production_metric < best_metric:
                return current_production_id
            else:
                return best_id
        else:
            if production_metric > best_metric:
                return current_production_id
            else:
                return best_id

    def _generate_artifact_path(self, run_id):
        return 'runs:/{}/{}'.format(run_id, self.model_name)

    def register_best_model(self, registration_message, log_db, log_table):
        best_id = self._find_best_model()
        try:
            current_production = self._get_current_production()
            current_production_version = self.get_production_version()
        except mlflow.exceptions.RestException:
            current_production = -1
            current_production_version = -1
        updated = current_production != best_id
        if updated:
            register_new_model = mlflow.register_model(self._generate_artifact_path(best_id),
                                                       self.experiment_name)
            self.client.update_registered_model(
                name = register_new_model.name, description = 'Total Passengers Forecast For 2015 Over Multiple Airports In NY')
            self.client.update_model_version(
                name = register_new_model.name, version = register_new_model.version,
                description = registration_message)
            self.client.transition_model_version_stage(
                name = register_new_model.name, version = register_new_model.version, stage = 'Production')
            if current_production_version > 0:
                self.client.transition_model_version_stage(
                    name = register_new_model.name, version = current_production_version, stage = 'Archived')
                RegistryLogging(log_db, log_table, self.experiment_title, int(register_new_model.version),
                                updated).write_entry()
                return 'Upgraded To Production'
        else:
            RegistryLogging(log_db, log_table, self.experiment_title, int(current_production_version),
                            updated).write_entry()
            return 'No Change'

    def get_model_as_udf(self):
        production_id = self._get_current_production()
        artifact_uri = self._generate_artifact_path(production_id)
        return mlflow.pyfunc.spark_udf(spark_session, model_uri = artifact_uri)

base_directory = '/Users/edwin.goh95@gmail.com/'
folder = 'Machine Learning Engineering In Action/'
logging_location = base_directory + folder + 'all_airports_logs'
# model_registry = ModelRegistration(
#     experiment_name = base_directory + folder + 'Total Passenger Forecasts For Multiple Airports',
#     experiment_title = 'Total Passengers Airport Forecasts',
#     model_name = 'Holt-Winters-Exponential-Smoothing', metric = 'best_trial_loss', direction = 'ASC')
# model_registry.register_best_model('Initial run', logging_location = logging_location,
#                                    log_db = 'edwingoh95', log_table = 'registry_status')

#%% Feature Stores
@dataclass
class SchemaTypes:
    string_columns: List[str]
    non_string_columns: List[str]

def get_column_types(df):
    strings = [x.name for x in df.schema if x.dataType == functions.StringType()]
    non_strings = [x for x in df.schema.names if x not in strings]
    return SchemaTypes(strings, non_strings)

def clean_messy_strings(df):
    columns = get_column_types(df)
    return df.select(*columns.non_string_columns, *[
        functions.regexp_replace(functions.col(x), ' ', '').alias(x) for x in columns.string_columns])

def fill_missing(df):
    columns = get_column_types(df)
    return df.select(*columns.non_string_columns, *[
        functions.when(functions.col(x) == '?', 'Unknown').otherwise(functions.col(x)).alias(x)
        for x in columns.string_columns])

def convert_label(df, label, true_condition_string):
    return df.withColumn(label, functions.when(functions.col(label) == true_condition_string, 1)
                         .otherwise(0))

def generate_features(df, id_augment):
    overtime = df.withColumn('Overtime', functions.when(functions.col('Hours_Worked_Per_Week') > 40,
                             1).otherwise(0))
    net_positive = overtime.withColumn('Gains', functions.when(
        functions.col('Capital Gain') > functions.col('Capital Loss'), 1).otherwise(0))
    high_education = net_positive.withColumn('Highly Educated', functions.when(
        functions.col('Education Years') >= 16, 2).when(functions.col('Education Years') > 12, 1)
        .otherwise(0))
    gender_key = high_education.withColumn('Gender_Key', functions.when(
        functions.col('Gender') == 'Female', 1).otherwise(0))
    primary_keys = gender_key.withColumn(
        'ID', functions.monotonically_increasing_id() + functions.lit(id_augment))
    return primary_keys

def data_augmentation(df, label, true_condition_label, id_augment = 0):
    clean_strings = clean_messy_strings(df)
    missing_filled = fill_missing(clean_strings)
    corrected_label = convert_label(missing_filled, label, true_condition_label)
    additional_features = generate_features(corrected_label, id_augment)
    return additional_features

# Feature Acquisition For Modelling
def generate_lookup(table, feature, key):
    return feature_store.FeatureLookup(table_name = table, feature_names = feature, lookup_key = key)