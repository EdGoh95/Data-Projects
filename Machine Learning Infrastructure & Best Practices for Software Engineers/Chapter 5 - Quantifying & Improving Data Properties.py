#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 5: Quantifying and Improving Data Properties
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from transformers import pipeline

#%% Basics of Feature Engineering for Text Data
vectorizer = CountVectorizer(max_features = 4)
sentence1 = 'fprintf("Hello World!");'
sentence2 = 'return 1'
X = vectorizer.fit_transform([sentence1, sentence2]).toarray()

BOW_df = pd.DataFrame(X, columns = vectorizer.get_feature_names_out(), index = [sentence1, sentence2])
print(BOW_df)

#%% Data Cleaning
#### Removing NA from the data
gerrit_reviews_df =  pd.read_csv("Source Code From GitHub/chapter_5/gerrit_reviews.csv", sep = ';')
print('Number of empty rows in Gerrit Reviews DataFrame (Before): {:d}'.format(
    gerrit_reviews_df['LOC'].isnull().sum()))
gerrit_reviews_df_cleaned = gerrit_reviews_df.dropna()
print('Number of empty rows in Gerrit Reviews DataFrame (After): {:d}'.format(
    gerrit_reviews_df_cleaned['LOC'].isnull().sum()))
cleaning_vectorizer = CountVectorizer(min_df = 2, max_df = 10)
gerrit_reviews_features_df = cleaning_vectorizer.fit_transform(gerrit_reviews_df_cleaned['LOC'])
gerrit_reviews_BOW_df = pd.DataFrame(gerrit_reviews_features_df.toarray(),
                                     columns = cleaning_vectorizer.get_feature_names_out(),
                                     index = gerrit_reviews_df_cleaned['LOC'])
print('Number of features in the Gerrit Review dataframe: {:d}'.format(
    len(gerrit_reviews_BOW_df.columns)))

#### Data Imputation
gerrit_reviews_dfNaNs = pd.read_csv("Source Code From GitHub/chapter_5/gerrit_reviews_nan.csv", sep = '$')
gerrit_reviews_NaNs_features = gerrit_reviews_dfNaNs.reset_index()
gerrit_reviews_NaNs_features.drop(['LOC', 'index'], axis = 1, inplace = True)
print('NaN values in the Gerrit Review DataFrame:', gerrit_reviews_NaNs_features.isnull().sum(), sep = '\n')
imputer = IterativeImputer(max_iter = 3, random_state = 42, verbose = 2)
# imputer.fit(gerrit_reviews_NaNs_features)
# gerrit_reviews_imputed = imputer.transform(gerrit_reviews_NaNs_features)
# gerrit_reviews_imputed_df = pd.DataFrame(gerrit_reviews_imputed)

#%% Noise Attribution
sentiment_pipeline = pipeline('sentiment-analysis')
sentiments_df = pd.DataFrame(sentiment_pipeline(list(gerrit_reviews_df_cleaned['message'])))
sentiments_df = sentiments_df.label.map({'NEGATIVE': 0, 'POSITIVE': 1})

# Train a RandomForest classifier to obtain the most important features
rf_clf = RandomForestClassifier(max_depth = 10, random_state = 42)
rf_clf.fit(gerrit_reviews_BOW_df, sentiments_df)

feature_importances_df = pd.DataFrame(rf_clf.feature_importances_, index = gerrit_reviews_BOW_df.columns,
                                      columns = ['Importance'])
feature_importances_df.sort_values(by = ['Importance'], ascending = False, inplace = True)
important_features_only_df = feature_importances_df[feature_importances_df['Importance'] != 0]
print('{:d} features in total, but only {:d} features were used in predictions'.format(
    feature_importances_df.shape[0], important_features_only_df.shape[0]))

plt.figure(figsize = (40, 20))
sns.barplot(y = important_features_only_df['Importance'][:30], x = important_features_only_df.index[:30],
            color = 'steelblue')
plt.title('Top 30 Features By Importance In Descending Order')
plt.xlabel('Features')
plt.xticks(rotation = 90)
plt.ylabel('Feature Importance')
plt.tight_layout()

#%% Data Splitting To Preserve Data Distribution
gerrit_reviews_BOW_train, gerrit_reviews_BOW_test, sentiments_train, sentiments_test = train_test_split(
    gerrit_reviews_BOW_df, sentiments_df, test_size = 0.33, random_state = 42)

fig, axes = plt.subplots(1, 2)
sns.histplot(gerrit_reviews_BOW_train['dataresponse'], binwidth = 0.2, ax = axes[0])
sns.histplot(gerrit_reviews_BOW_test['dataresponse'], binwidth = 0.2, ax = axes[1])

gerrit_reviews_BOW_feature_distribution_train = gerrit_reviews_BOW_train.groupby(by = 'dataresponse').count()
gerrit_reviews_BOW_feature_distribution_test = gerrit_reviews_BOW_test.groupby(by = 'dataresponse').count()

fig, axes = plt.subplots(1, 2)
sns.histplot(sentiments_train, binwidth = 0.5, ax = axes[0])
sns.histplot(sentiments_test, binwidth = 0.5, ax = axes[1])