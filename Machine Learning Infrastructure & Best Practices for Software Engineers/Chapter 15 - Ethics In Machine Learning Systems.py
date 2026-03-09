#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 15: Ethics in Machine Learning Systems
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#%% Mock Salary Dataset
salary_df = pd.DataFrame({'Age': [25, 45, 35, 50, 23, 30, 40, 28, 38, 48, 27, 37, 47, 26, 36, 46],
                          'Income': [50000, 100000, 75000, 120000, 45000, 55000, 95000, 65000, 85000,
                                     110000, 48000, 58000, 98000, 68000, 88000, 105000],
                          'Gender': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], # 1: Male, 0: Female
                          'Hired': [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1] # 1: Hired, 0: Not Hired
                          })

salary_train_df, salary_test_df = train_test_split(salary_df, test_size = 0.2, random_state = 42,
                                                   stratify = salary_df['Gender'])

# Converting the dataframes into BinaryLabelDataset format as required by the IBM  AI Fairness 360 framework
salary_train_bld = BinaryLabelDataset(df = salary_train_df, label_names = ['Hired'],
                                      protected_attribute_names = ['Gender'])
salary_test_bld = BinaryLabelDataset(df = salary_test_df, label_names = ['Hired'],
                                     protected_attribute_names = ['Gender'])

# Compute the fairness metric on the training dataset
salary_train_bld_metric = BinaryLabelDatasetMetric(
    dataset = salary_train_bld, unprivileged_groups = [{'Gender': 1}], privileged_groups = [{'Gender': 0}])
print('Salary Binary Label Dataset Disparity: {:.3f}'.format(salary_train_bld_metric.mean_difference()))

# Mitigate bias by reweighing the dataset
reweighted = Reweighing(unprivileged_groups = [{'Gender': 1}], privileged_groups = [{'Gender': 0}])
salary_train_bld_reweighted = reweighted.fit_transform(salary_train_bld)

# Compute the fairness metric on the reweighted training dataset
salary_train_bld_reweighted_metric = BinaryLabelDatasetMetric(dataset = salary_train_bld_reweighted,
                                                              unprivileged_groups = [{'Gender': 1}],
                                                              privileged_groups = [{'Gender': 0}])
print('Re-Weighted Salary Binary Label Datset Disparity: {:.3f}'.format(
    salary_train_bld_reweighted_metric.mean_difference()))

#%% Titanic Dataset
titanic_df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")

# Data Pre-processing
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 1, 'female': 0}) # Converting the entries in the 'Sex' column to binary
titanic_df.drop(['Name'], axis = 1, inplace = True)
titanic_train_df, titanic_test_df = train_test_split(titanic_df, test_size = 0.2, random_state = 42,
                                                     stratify = titanic_df['Sex'])

# Converting the dataframes into BinaryLabelDatasets
titanic_train_bld = BinaryLabelDataset(df = titanic_train_df, label_names = ['Survived'],
                                       protected_attribute_names = ['Sex'])
titanic_test_bld = BinaryLabelDataset(df = titanic_test_df, label_names = ['Survived'],
                                      protected_attribute_names = ['Sex'])

# Compute the fairness metric on the training dataset
titanic_train_bld_metric = BinaryLabelDatasetMetric(dataset = titanic_train_bld,
                                                    unprivileged_groups = [{'Sex': 0}],
                                                    privileged_groups = [{'Sex': 1}])
print('Titanic Binary Label Dataset Disparity: {:.3f}'.format(titanic_train_bld_metric.mean_difference()))

# Mitigate bias by reweighing the dataset
titanic_reweighted = Reweighing(unprivileged_groups = [{'Sex': 0}], privileged_groups = [{'Sex': 1}])
titanic_train_bld_reweighted = titanic_reweighted.fit_transform(titanic_train_bld)

# Compute the fairness metric on the reweighted training dataset
titanic_train_bld_reweighted_metric = BinaryLabelDatasetMetric(dataset = titanic_train_bld_reweighted,
                                                               unprivileged_groups = [{'Sex': 0}],
                                                               privileged_groups = [{'Sex': 1}])
print('Re-Weighted Titanic Binary Label Dataset Disparity: {:.3f}'.format(
    titanic_train_bld_reweighted_metric.mean_difference()))

titanic_scaler = StandardScaler()
titanic_features_train = titanic_scaler.fit_transform(titanic_train_bld_reweighted.features)
titanic_target_train = titanic_train_bld_reweighted.labels.ravel()
titanic_classifier = LogisticRegression().fit(titanic_features_train, titanic_target_train)

titanic_features_test = titanic_scaler.transform(titanic_test_bld.features)
titanic_target_test = titanic_test_bld.labels.ravel()
titanic_predictions = titanic_classifier.predict(titanic_features_test)

print('Test Accuracy: {:.4f}'.format(accuracy_score(titanic_target_test, titanic_predictions)))
print('Test Classification Report:', classification_report(
    titanic_target_test, titanic_predictions, target_names = ['Did Not Survived', 'Survived']), sep = '\n')