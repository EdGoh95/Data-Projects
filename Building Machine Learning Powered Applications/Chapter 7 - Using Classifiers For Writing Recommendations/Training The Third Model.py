#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 7: Using Classifiers for Writing Recommendations
"""
import sys
if '../' not in sys.path:
    sys.path.append('../')
import pandas as pd
import numpy as np
import joblib
from helper_functions import POS, get_split_by_author, get_metrics, plot_confusion_matrix, \
    plot_ROC_curve, plot_calibration_curve, get_feature_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from termcolor import colored

#%% Data Preparation
posts_features_df = pd.read_csv('../Data Science Posts With Features.csv')
questions_df = posts_features_df[posts_features_df['is_question']]
train_df, test_df = get_split_by_author(questions_df.copy(), test_size = 0.2)

feature_columns = ['question_mark_full', 'question_word_full', 'action_verb_full', 'normalised_text_length',
                   'num_words', 'num_unique_words', 'num_stop_words', 'average_word_length',
                   'num_question_marks', 'num_full_stops', 'num_commas', 'num_exclamation_marks',
                   'num_colons', 'num_semicolons', 'num_quotes', 'polarity'] + POS

train_features = train_df[feature_columns].astype(float)
train_labels = train_df['Score'] > train_df['Score'].median()
test_features = test_df[feature_columns].astype(float)
test_labels = test_df['Score'] > test_df['Score'].median()

#%% Model Training & Tuning
# randforest_classifier = RandomForestClassifier(class_weight = 'balanced', oob_score = True,
#                                                 n_jobs = -1, verbose = 1)
# hyperparam_grid = {'n_estimators': np.linspace(start = 100, stop = 1000, num = 19, dtype = int),
#                     'max_depth': np.linspace(start = 5, stop = 50, num = 10, dtype = int)}
# randforest_with_GridSearch = GridSearchCV(estimator = randforest_classifier, param_grid = hyperparam_grid,
#                                           scoring = 'average_precision', n_jobs = -1, verbose = 2)
# randforest_with_GridSearch.fit(train_features, train_labels)
# tuned_model = randforest_with_GridSearch.best_estimator_
# joblib.dump(tuned_model, '../Models/Model_v3.pkl') # Saving the trained model

tuned_model = joblib.load('../Models/Model_v3.pkl')
predicted_labels = tuned_model.predict(test_features)
predicted_probabilities = tuned_model.predict_proba(test_features)

#%% Model Evaluation
predictions = np.argmax(tuned_model.oob_decision_function_, axis = 1)
train_accuracy, train_precision, train_recall, train_F1 = get_metrics(train_labels, predictions)
print('Training - Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}'.format(
    train_accuracy, train_precision, train_recall, train_F1))

validation_accuracy, validation_precision, validation_recall, validation_F1 = get_metrics(
    test_labels, predicted_labels)
print('Validation - Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}'.format(
    validation_accuracy, validation_precision, validation_recall, validation_F1))

plot_confusion_matrix(test_labels, predicted_labels)
plot_ROC_curve(test_labels, predicted_probabilities[:, 1], FPR_line = 0.1)
plot_calibration_curve(test_labels, predicted_probabilities[:, 1])

# Feature Importance
k = 20

print(colored('{:d} Most Important Features:'.format(k), 'green', attrs = ['bold']))
for entry in get_feature_importance(tuned_model, np.array(feature_columns))[:k]:
    print('{}: {:.2g}'.format(entry[0], entry[1]))