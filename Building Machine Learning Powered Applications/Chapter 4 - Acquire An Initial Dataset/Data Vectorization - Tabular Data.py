#!/usr/bin/env python3
"""
Building Machine Learning Powered Applications Chapter 4: Acquire an Initial Dataset
"""
import pandas as pd

datascience_posts_df = pd.read_csv("../Data Science Posts.csv").set_index('Id')
datascience_posts_df['is_question'] = datascience_posts_df['PostTypeId'] == 1
tabular_df = datascience_posts_df[datascience_posts_df['is_question']][['Tags', 'CommentCount', 'CreationDate', 'Score']]

#%% Tabular Data
def get_norm(df, col):
    return (df[col] - df[col].mean())/df[col].std()

tabular_df['NormalisedComment'] = get_norm(tabular_df, 'CommentCount')
tabular_df['NormalisedScore'] = get_norm(tabular_df, 'Score')

# Converting the CreationDate to a pandas datetime
tabular_df['Date'] = pd.to_datetime(tabular_df['CreationDate'])
# Extracting meaningful features from the datetime column
tabular_df['Year'] = tabular_df['Date'].dt.year
tabular_df['Month'] = tabular_df['Date'].dt.month
tabular_df['Day'] = tabular_df['Date'].dt.day
tabular_df['Hour'] = tabular_df['Date'].dt.hour

# Extract tags and transform them into arrays of tags
clean_tags = tabular_df['Tags'].str.split('|').apply(lambda x: [a.strip('|').strip('|') for a in x])
# Select only tags that appear more than 500 times
tag_columns = pd.get_dummies(clean_tags.apply(pd.Series).stack()).groupby(level = 0).sum()
all_tags = tag_columns.astype(bool).sum(axis = 0).sort_values(ascending = False).drop('')
top_tags = tag_columns[all_tags[all_tags > 1000].index]
# Add the top tags back into tabular_df and retaining the vectorized features
vectorized_columns = ['Year', 'Month', 'Day', 'NormalisedComment', 'NormalisedScore'] + list(top_tags.columns)
features_df = pd.concat([tabular_df, top_tags], axis = 1)[vectorized_columns]