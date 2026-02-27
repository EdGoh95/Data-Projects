#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 5: Train and Evaluate Your Model
"""
import sys
if '../' not in sys.path:
    sys.path.append('../')
import pandas as pd
import numpy as np
import joblib
from helper_functions import format_raw_df, add_features, get_split_by_author, \
    get_vectorised_series, get_feature_vectors_and_labels, get_metrics, \
        plot_confusion_matrix, plot_ROC_curve, plot_calibration_curve, get_feature_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from termcolor import colored

#%% Data Preparation, Pre-processing & Splitting
datascience_posts_df = pd.read_csv('../Data Science Posts.csv')
formatted_posts_df = format_raw_df(datascience_posts_df.copy())
questions_df = formatted_posts_df[formatted_posts_df['is_question']]
processed_df = add_features(questions_df.copy())
train_df, test_df = get_split_by_author(processed_df.copy(), test_size = 0.2)

#%% Data Vectorisation
# vectoriser = TfidfVectorizer(strip_accents = 'ascii', min_df = 5, max_df = 0.5, max_features = 10000)
# vectoriser.fit(train_df['full_text'])
# joblib.dump(vectoriser, '../Models/Vectoriser_v1.pkl') # Saving the fitted vectoriser

vectoriser = joblib.load('../Models/Vectoriser_v1.pkl')
train_df['vectors'] = get_vectorised_series(train_df['full_text'], vectoriser)
test_df['vectors'] = get_vectorised_series(test_df['full_text'], vectoriser)

feature_columns = ['question_mark_full', 'question_word_full', 'action_verb_full', 'normalised_text_length']
train_features, train_labels = get_feature_vectors_and_labels(train_df, feature_columns)
test_features, test_labels = get_feature_vectors_and_labels(test_df, feature_columns)

#%% Model Training & Evaluation
# randforest_classifier = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced',
#                                                 oob_score = True)
# randforest_classifier.fit(train_features, train_labels)
# joblib.dump(randforest_classifier, '../Models/Model_v1.pkl') # Saving the trained model

randforest_classifier = joblib.load('../Models/Model_v1.pkl')
predicted_labels = randforest_classifier.predict(test_features)
predicted_probabilities = randforest_classifier.predict_proba(test_features)

predictions = np.argmax(randforest_classifier.oob_decision_function_, axis = 1)
train_accuracy, train_precision, train_recall, train_F1 = get_metrics(train_labels, predictions)
print('Training - Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}'.format(
    train_accuracy, train_precision, train_recall, train_F1))

validation_accuracy, validation_precision, validation_recall, validation_F1 = get_metrics(
    test_labels, predicted_labels)
print('Validation - Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}'.format(
    validation_accuracy, validation_precision, validation_recall, validation_F1))

plot_confusion_matrix(test_labels, predicted_labels)
# plot_ROC_curve(test_labels, predicted_probabilities[:, 1])
plot_ROC_curve(test_labels, predicted_probabilities[:, 1], FPR_line = 0.1)
plot_calibration_curve(test_labels, predicted_probabilities[:, 1])

# Feature Importance
k = 10
features = np.append(vectoriser.get_feature_names_out(), feature_columns)

print(colored('{:d} Most Important Features:'.format(k), 'yellow', attrs = ['bold']))
for entry in get_feature_importance(randforest_classifier, features)[:k]:
    print('{}: {:.2g}'.format(entry[0], entry[1]))

print(colored('{:d} Least Important Features:'.format(k), 'cyan', attrs = ['bold']))
for entry in get_feature_importance(randforest_classifier, features)[-k:]:
    print('{}: {:.2g}'.format(entry[0], entry[1]))