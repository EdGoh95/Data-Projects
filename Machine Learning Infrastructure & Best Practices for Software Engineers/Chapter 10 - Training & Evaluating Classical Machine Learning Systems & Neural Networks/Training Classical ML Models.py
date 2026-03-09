#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 10: Training and Evaluating Classical Machine Learning Systems and Neural Networks
"""
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

#%% Basic Decision Tree Model
ant_df = pd.read_excel(
    "../Chapter 6 - Processing Data In Machine Learning Systems/Regenerated PROMISE and BPD Datasets.xlsx",
    sheet_name = 'ant_1_3', index_col = 0)
ant_features = ant_df.drop(['Defect'], axis = 1)
ant_target = ant_df['Defect']
ant_features_train, ant_features_test, ant_target_train, ant_target_test = train_test_split(
    ant_features, ant_target, train_size = 0.9, random_state = 42)

ant_decision_tree_model = DecisionTreeClassifier()
ant_decision_tree_model.fit(ant_features_train, ant_target_train)
ant_prediction = ant_decision_tree_model.predict(ant_features_test)

joblib.dump(ant_decision_tree_model,
            '../Source Code From GitHub/chapter_12/chapter_12_decision_tree_model.joblib' )

print(colored('ant_1_3', 'blue', attrs = ['bold']))
print('Test Accuracy: {:.3f}'.format(accuracy_score(ant_target_test, ant_prediction)), sep = '\n')
print('Test Precision: {:.3f}'.format(
    precision_score(ant_target_test, ant_prediction, average = 'weighted')), sep = '\n')
print('Test Recall: {:.3f}'.format(
    recall_score(ant_target_test, ant_prediction, average = 'weighted')), sep = '\n')

# export_graphviz(decision_tree = ant_decision_tree_model, feature_names = list(ant_features_train.columns),
#                 filled = True, out_file = 'ant_1_3 Decision Tree.dot')

camel_df = pd.read_excel(
    "../Chapter 6 - Processing Data In Machine Learning Systems/Regenerated PROMISE and BPD Datasets.xlsx",
    sheet_name = 'camel_1_2', index_col = 0)
camel_features = camel_df.drop(['Defect'], axis = 1)
camel_target = camel_df['Defect']
camel_features_train, camel_features_test, camel_target_train, camel_target_test = train_test_split(
    camel_features, camel_target, train_size = 0.9, random_state = 42)

camel_decision_tree_model = DecisionTreeClassifier()
camel_decision_tree_model.fit(camel_features_train, camel_target_train)
camel_prediction = camel_decision_tree_model.predict(camel_features_test)

print(colored('\ncamel_1_3', 'green', attrs = ['bold']))
print('Test Accuracy: {:.3f}'.format(accuracy_score(camel_target_test, camel_prediction)),
      sep = '\n')
print('Test Precision: {:.3f}'.format(
    precision_score(camel_target_test, camel_prediction, average = 'weighted')), sep = '\n')
print('Test Recall: {:.3f}'.format(
    recall_score(camel_target_test, camel_prediction, average = 'weighted')), sep = '\n')

# export_graphviz(decision_tree = camel_decision_tree_model, feature_names = list(camel_features_train.columns),
#                 filled = True, out_file = 'camel_1_2 Decision Tree.dot')

#%% Ensemble Models (Example: RandomForest)
camel_randforest_model = RandomForestClassifier()
camel_randforest_model.fit(camel_features_train, camel_target_train)
camel_randforest_prediction = camel_randforest_model.predict(camel_features_test)

print(colored('\ncamel_1_3 (RandomForest)', 'red', attrs = ['bold']))
print('Test Accuracy: {:.3f}'.format(accuracy_score(camel_target_test, camel_randforest_prediction)),
      sep = '\n')
print('Test Precision: {:.3f}'.format(
    precision_score(camel_target_test, camel_randforest_prediction, average = 'weighted')), sep = '\n')
print('Test Recall: {:.3f}'.format(
    recall_score(camel_target_test, camel_randforest_prediction, average = 'weighted')), sep = '\n')

camel_feature_importance_df = pd.DataFrame(camel_randforest_model.feature_importances_,
                                           index = camel_features.columns, columns = ['Importance'])
camel_feature_importance_df.sort_values(by = ['Importance'], ascending = False, inplace = True)
camel_important_features_df = camel_feature_importance_df[camel_feature_importance_df['Importance'] != 0]
print('{:d} features in total, but only {:d} contributed to predictions.'.format(
    camel_feature_importance_df.shape[0], camel_important_features_df.shape[0]))

plt.figure(figsize = (40, 10))
sns.barplot(x = camel_important_features_df.index, y = camel_important_features_df['Importance'],
            color = 'steelblue')
plt.title('Feature Importance In Descending Order')
plt.xlabel('Features', fontsize = 14)
plt.ylabel('Feature Importance', fontsize = 14)
plt.tight_layout()