#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 10: Build Safeguards for Models
(For Batch Deployments)
"""
import numpy as np
import json
heuristics = __import__('Chapter 3 - Build Your First End-To-End Pipeline')
from sklearn.ensemble import RandomForestClassifier

required_features = {'is_question': bool, 'question_mark_full': bool, 'question_word_full': bool,
                     'action_verb_full': bool, 'normalised_text_length': float}
with open('Features Mapping.json') as json_file:
    features_mapping = json.load(json_file)
feature_columns = list(features_mapping.keys())

def check_missing_features(questions_df):
    '''
    Checks whether any of the required features are missing in the dataframe
    params:
        questions_df - Dataframe containing questions to be evaluated
    returns:
        List containing the required features that are not present in the dataframe
    '''
    missing = []
    for feature in required_features.keys():
        if feature not in questions_df.columns:
            missing.append(feature)
    return missing

def check_feature_data_types(questions_df):
    '''
    Checks whether the data types of the required features in the dataframe are as expected
    params:
        questions_df - Dataframe containing questions to be evaluated
    returns:
        List containing features that are of the incorrect data types
    '''
    incorrect_data_types = []
    for feature, data_type in required_features.items():
        if questions_df[feature].dtype != data_type:
            incorrect_data_types.append((questions_df[feature], data_type))
    return incorrect_data_types

def check_output_type_and_range(question_score):
    '''
    Checks whether the output type and range are as expected
    params:
        question_score - Model output
    '''
    if not isinstance(question_score, float):
        raise ValueError('Wrong output type: {} (Expected to be a float instead of {})'.format(
            question_score, type(question_score).__name__))
    if not 0.0 <= question_score <= 1.0:
        raise ValueError('Output out of range: {} (Expected to between 0 and 1)'.format(question_score))

def run_heuristic(question_length):
    '''
    A function stub (template)
    '''
    pass

def run_model(questions_df):
    '''
    A function stub (template)
    params:
        questions_df - Dataframe containing questions to be evaluated
    '''
    # Insert any slow model inference here
    pass

def get_filtering_model(classifier, features, labels):
    '''
    Trains a model to filter out examples that a pre-trained binary classification model struggles with
    (incorrectly classified examples)
    params:
        classifier - Any pre-trained binary classification model
        features - Subset of features used to train the classifier
        labels - Actual labels
    Returns:
        A filtering model that is able to identify examples where the classifier struggles on
    '''
    predictions = classifier.predict(features)
    is_error = [prediction != ground_truth for prediction, ground_truth in zip(predictions, labels)]
    filtering_model = RandomForestClassifier(class_weight = 'balanced', oob_score = True, n_jobs = -1)
    filtering_model.fit(features, is_error)
    return filtering_model

def validate_and_handle_request(questions_df):
    '''
    Checks whether the given dataframe is missing any of the expected features and their data types are
    as expected
    params:
        questions_df - Dataframe containing questions to be evaluated in terms of quality
    '''
    missing_features = check_missing_features(questions_df)
    if len(missing_features) > 0:
        raise ValueError('Missing feature(s): {}'.format(missing_features))

    incorrect_data_types = check_feature_data_types(questions_df)
    if len(incorrect_data_types) > 0:
        if 'text_length' in questions_df.keys():
            if questions_df['text_length'].dtype == int:
                return run_heuristic(questions_df['text_length'])
        raise ValueError('Incorrect data type(s): {}'.format(incorrect_data_types))

    return run_model(questions_df)

def validate_and_correct_output(question_score, questions_df):
    '''
    Verify whether the model output type and score are as expected. If not, run a heuristic by
    executing the run_heuristic function
    params:
        question_score - Model output
        questions_df - Dataframe containing questions to be evaluated
    '''
    try:
        check_output_type_and_range(question_score)
        return question_score
    except ValueError:
        run_heuristic(questions_df['text_length'])