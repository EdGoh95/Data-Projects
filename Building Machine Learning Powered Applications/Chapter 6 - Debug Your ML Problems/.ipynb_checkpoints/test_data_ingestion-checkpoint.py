#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 6: Debug Your ML Problems
"""
import sys
if '../' not in sys.path:
    sys.path.append('../')
import pandas as pd
from helper_functions import parse_xml_to_csv

REQUIRED_COLUMNS = ['Id', 'PostTypeId', 'Score', 'Body', 'Title', 'AnswerCount', 'AcceptedAnswerId',
                    'body_text']
ACCEPTED_TEXT_LENGTH_MEANS = pd.Interval(left = 20, right = 2000)

def get_fixture_df():
    '''
    Uses parser to return the MiniPosts dataframe
    returns:
        MiniPosts dataframe
    '''
    return parse_xml_to_csv('../Source Code From GitHub/tests/fixtures/MiniPosts.xml')

def test_parser_returns_dataframe():
    '''
    Tests that the parser works as expected and returns a DataFrame
    '''
    df = get_fixture_df()
    assert isinstance(df, pd.DataFrame)

def test_feature_columns_exist():
    '''
    Validates that all the required columns are present
    '''
    df = get_fixture_df()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns

def test_features_not_all_nulls():
    '''
    Validate that none of the features have a missing value
    '''
    df = get_fixture_df()
    for col in REQUIRED_COLUMNS:
        assert not df[col].isnull().all()

def test_text_mean():
    '''
    Validate that text mean matches expectations from initial data exploration (EDA)
    '''
    df = get_fixture_df()
    df['text_length'] = df['body_text'].str.len()
    text_col_mean = df['text_length'].mean()
    assert text_col_mean in ACCEPTED_TEXT_LENGTH_MEANS