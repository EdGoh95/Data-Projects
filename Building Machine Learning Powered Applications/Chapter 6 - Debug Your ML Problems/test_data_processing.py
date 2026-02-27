#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 6: Debug Your ML Problems
"""
import pytest
import sys
if '../' not in sys.path:
    sys.path.append('../')
import pandas as pd
from helper_functions import format_raw_df, add_features

REQUIRED_FEATURES = ['is_question', 'question_mark_full', 'action_verb_full', 'text_length',
                     'normalised_text_length']

@pytest.fixture
def df_with_features():
    df = pd.read_csv('../Data Science Posts.csv')
    df = format_raw_df(df.copy())
    return add_features(df.copy())

def test_feature_presence(df_with_features):
    for feature in REQUIRED_FEATURES:
        assert feature in df_with_features.columns

def test_feature_type(df_with_features):
    assert df_with_features['is_question'].dtype == bool
    assert df_with_features['question_mark_full'].dtype == bool
    assert df_with_features['action_verb_full'].dtype == bool
    assert df_with_features['text_length'].dtype == int
    assert df_with_features['normalised_text_length'].dtype == float

def test_text_length(df_with_features):
    text_mean = df_with_features['text_length'].mean()
    text_max = df_with_features['text_length'].max()
    text_min = df_with_features['text_length'].min()
    assert text_mean in pd.Interval(left = 20, right = 2000)
    assert text_max in pd.Interval(left = 0, right = 35000)
    assert text_min in pd.Interval(left = 0, right = 1000)