#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 5: Train and Evaluate Your Model
"""
import sys
if '../' not in sys.path:
    sys.path.append('../')
import pandas as pd
from helper_functions import format_raw_df, get_randomised_train_test_split, get_split_by_author
from termcolor import colored

datascience_posts_df = pd.read_csv('../Data Science Posts.csv')
formatted_posts_df = format_raw_df(datascience_posts_df.copy())
questions_df = formatted_posts_df[formatted_posts_df['is_question']]

#%% Using Randomised Split
train_questions_df_rand, test_questions_df_rand = get_randomised_train_test_split(questions_df)
print(colored('Randomised Split:', 'green', attrs = ['bold']))
print('{:d} questions in the training set, and {:d} questions in the test set'.format(
    len(train_questions_df_rand), len(test_questions_df_rand)))
train_questions_owners_rand = set(train_questions_df_rand['OwnerUserId'].values)
test_questions_owners_rand = set(test_questions_df_rand['OwnerUserId'].values)
print('{:d} different owners in the training set, {:d} different owners in the test set'.format(
    len(train_questions_owners_rand), len(test_questions_owners_rand)))
print('{:d} owners appear in both the training and the test sets.'.format(
    len(train_questions_owners_rand.intersection(test_questions_owners_rand))))

#%% Using Grouped By Author Split
train_questions_df_by_owner, test_questions_df_by_owner = get_split_by_author(questions_df)
print(colored('\nSplit By Author:', 'blue', attrs = ['bold']))
print('{:d} questions in the training set, and {:d} questions in the test set'.format(
    len(train_questions_df_by_owner), len(test_questions_df_by_owner)))
train_questions_owners = set(train_questions_df_by_owner['OwnerUserId'].values)
test_questions_owners = set(test_questions_df_by_owner['OwnerUserId'].values)
print('{:d} different owners in the training set, {:d} different owners in the test set'.format(
    len(train_questions_owners), len(test_questions_owners)))
print('{:d} owners appear in both the training and the test sets.'.format(
    len(train_questions_owners.intersection(test_questions_owners))))