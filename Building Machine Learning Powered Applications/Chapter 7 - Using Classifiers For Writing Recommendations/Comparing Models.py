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
from helper_functions import get_split_by_author, POS, get_vectorised_series, \
    get_feature_vectors_and_labels, get_metrics, get_feature_importance, plot_multiple_calibration_curves
from termcolor import colored

#%% Data Preparation
posts_features_df = pd.read_csv('../Data Science Posts With Features.csv')
questions_df = posts_features_df[posts_features_df['is_question']]
train_df, test_df = get_split_by_author(questions_df.copy(), test_size = 0.2)

first_feature_set = ['question_mark_full', 'question_word_full', 'action_verb_full', 'normalised_text_length']
second_feature_set = ['question_mark_full', 'question_word_full', 'action_verb_full', 'normalised_text_length',
                      'num_words', 'num_unique_words', 'num_stop_words', 'average_word_length',
                      'num_question_marks', 'num_full_stops', 'num_commas', 'num_exclamation_marks',
                      'num_colons', 'num_semicolons', 'num_quotes', 'polarity'] + POS

#%% Data Vectorisation
first_vectoriser = joblib.load('../Models/Vectoriser_v1.pkl')
second_vectoriser = joblib.load('../Models/Vectoriser_v2.pkl')
test_df['vectors'] = get_vectorised_series(test_df['full_text'].copy(), second_vectoriser)

test_features_1, test_labels_1 = get_feature_vectors_and_labels(test_df, first_feature_set)
test_features_2, test_labels_2 = get_feature_vectors_and_labels(test_df, second_feature_set)
test_features_3 = test_df[second_feature_set].astype(float)
test_labels_3 = test_df['Score'] > test_df['Score'].median()

#%% Loading The Models
model1 = joblib.load('../Models/Model_v1.pkl')
model2 = joblib.load('../Models/Model_v2.pkl')
model3 = joblib.load('../Models/Model_v3.pkl')
models = [model1, model2, model3]

#%% Comparing Model Evaluations
predicted_labels_1 = model1.predict(test_features_1)
predicted_probabilities_1 = model1.predict_proba(test_features_1)
model1_features = np.append(first_vectoriser.get_feature_names_out(), first_feature_set)

predicted_labels_2 = model2.predict(test_features_2)
predicted_probabilities_2 = model2.predict_proba(test_features_2)
model2_features = np.append(second_vectoriser.get_feature_names_out(), second_feature_set)

predicted_labels_3 = model3.predict(test_features_3)
predicted_probabilities_3 = model3.predict_proba(test_features_3)

test_labels_combined = [test_labels_1, test_labels_2, test_labels_3]
predicted_labels_combined = [predicted_labels_1, predicted_labels_2, predicted_labels_3]
predicted_probabilities_combined = [
    predicted_probabilities_1[:, 1], predicted_probabilities_2[:, 1], predicted_probabilities_3[:, 1]]
features_combined = [model1_features, model2_features, np.array(second_feature_set)]
colours = ['red', 'magenta', 'blue', 'cyan', 'green', 'yellow']
k = 10

for index, (test_labels, predicted_labels, model, features) in enumerate(zip(
        test_labels_combined, predicted_labels_combined, models, features_combined)):
    print(colored('Model {:d}:'.format(index + 1), colours[index * 2], attrs = ['bold']), end = ' ')
    accuracy, precision, recall, F1 = get_metrics(test_labels, predicted_labels)
    print('Validation - Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}'.format(
        accuracy, precision, recall, F1))

    print(colored('{:d} Most Important Features'.format(k), colours[(index * 2) + 1], attrs = ['bold']))
    for entry in get_feature_importance(model, features)[:k]:
        print('{}: {:.2g}'.format(entry[0], entry[1]))
    print()

plot_multiple_calibration_curves(test_labels_combined, predicted_probabilities_combined)