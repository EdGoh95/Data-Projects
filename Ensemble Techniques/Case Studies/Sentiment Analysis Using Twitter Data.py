#!/usr/bin/env python3
"""
Hands-On Ensemble Learning with Python (Packt Publishing) Chapter 11:
Evaluating Sentiment on Twitter
"""
import time
import os
import json
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import re
import matplotlib.pyplot as plt
from termcolor import colored
colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'grey', 'white']
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# StreamListener present in tweepy version 3.10.0 and below
from tweepy import OAuthHandler, Stream, StreamListener

start = time.perf_counter()
#%% Load The Dataset Containing Sentiment Labelling
labels = ['Polarity', 'ID', 'Date', 'Query', 'Username', 'Tweet']
sentiment_df = pd.read_csv('Data/Sentiment140.csv', names = labels, 
                           encoding = 'latin-1')

tweet_sentiment_df = sentiment_df[['Tweet', 'Polarity']]
tweet_sentiment_df['Polarity'].replace(4, 1, inplace = True)

#%% Data Exploration and Analysis
sentiment_df.groupby('Polarity')['ID'].count().plot(
    kind = 'bar', title = 'Distribution of Tweet Polarities', xlabel = 'Count')

sentiment_df['Words'] = sentiment_df['Tweet'].str.split()
words = []
for tweet in sentiment_df['Words'].to_list():
    for word in tweet:
        words.append(word)
most_frequent = Counter(words).most_common(30)

plt.figure()
plt.plot(*zip(*most_frequent))
plt.title('30 Most Frequent Words')
plt.xticks(rotation = 60)
plt.ylabel('Count')

#%% Data Cleaning and Pro-Processing
# Include a list of stop words
stop_words = stopwords.words('english')
quotes_removed = []
for word in stop_words:
    if "'" in word:
        quotes_removed.append(re.sub(r'\'', '', word))
stop_words.append(quotes_removed)
    
def clean_string(string):
    '''
    Removes characters such as references (@), hashtags (#), URL links and HTML
    attributes since these do not contribute to the Tweet's sentiment
    '''
    # Remove HTML attributes (starting with &)
    modified_string = re.sub(r'\&\w*;', '', string)
    # Remove user mentions (@)
    modified_string = re.sub(r'@(\w+)', '', modified_string)
    # Remove links
    modified_string = re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+', '', 
                              modified_string)
    # Remove hashtags
    modified_string = re.sub(r'#(\w+)', '', modified_string)
    # Remove repeating characters
    modified_string = re.sub(r'(.)\1{1,}', r'\1\1', modified_string)
    # Remove anything that is not a letter
    modified_string = re.sub('[^a-zA-Z]', ' ', modified_string)
    # Remove anything with less than 2 characters
    modified_string = re.sub(r'\b\w{1,2}\b', '', modified_string)
    # Set all characters in the string to lowercase
    modified_string = modified_string.lower()
    # Remove multiple spaces
    modified_string = re.sub(r'\s\s+', ' ', modified_string)
    return modified_string

def preprocessing(string):
    '''
    Removes any punctuations and stopwords
    '''
    punctuations_removed = ''.join([char for char in string 
                                    if char not in punctuation])
    preprocessed_string = []
    for word in punctuations_removed.split(' '):
        if word not in stop_words:
            preprocessed_string.append(
                PorterStemmer().stem(word, to_lowercase = True))
    return ' '.join(preprocessed_string)

preprocessed_data = tweet_sentiment_df.sample(frac = 1).reset_index(drop = True)
preprocessed_data['Tweet'] = preprocessed_data['Tweet'].apply(clean_string)
preprocessed_data['Tweet'] = preprocessed_data['Tweet'].apply(preprocessing)
# preprocessed_data.to_csv('Data/Sentiment140 (Pre-Processed).csv', index = False)

#%% Sentiment Analysis Performance Evaluation
def check_features_ngrams(features, n_grams, classifiers, colours):
    print('Features: {}, n_grams range: {}'.format(features, n_grams))
    # Instantiate the Inverse Document Frequency (IDF) feature extractor
    train_size = 10000
    test_start = 10000
    test_end = 100000
    tf_idf = TfidfVectorizer(max_features = features, ngram_range = n_grams,
                             stop_words = 'english')
    tf_idf.fit(preprocessed_data['Tweet'])
    tf_idf_transformed = tf_idf.transform(preprocessed_data['Tweet'])
    
    np.random.seed(123456)
    def check_classifier(name, classifier, colour):
        print(colored(name, colour, attrs = ['bold']))
        train_features = tf_idf_transformed[:train_size]
        train_target = preprocessed_data['Polarity'][:train_size].values
        classifier.fit(train_features, train_target)
        training_accuracy = accuracy_score(train_target, 
                                           classifier.predict(train_features))
        
        test_features = tf_idf_transformed[test_start:test_end]
        test_target = preprocessed_data['Polarity'][test_start:test_end].values
        testing_accuracy = accuracy_score(test_target, 
                                          classifier.predict(test_features))
        
        classifier_df = pd.DataFrame({'Features': [features], 
                                      'n_grams': [n_grams[-1]],
                                      'Classifier': [name], 
                                      'Training Accuracy': [training_accuracy],
                                      'Testing Accuracy': [testing_accuracy]})
        classifier_df = classifier_df.round(decimals = 4)
        if not os.path.isfile('Data/Sentiment Analysis Performance Evaluation.csv'):
            classifier_df.to_csv('Data/Sentiment Analysis Performance Evaluation.csv', 
                                 index = False)
        else:
            classifier_df.to_csv('Data/Sentiment Analysis Performance Evaluation.csv', 
                                 mode = 'a', header = False, index = False)
            
    for (name, classifier), colour in zip(classifiers, colours):
        check_classifier(name, classifier, colour)

# Sample all possible features and n-grams combinations
n_features = [500, 1000, 5000, 10000, 20000, 30000]
n_grams_range = [(1, 1), (1, 2), (1, 3)]
base_learners = [('Logistic Regression', LogisticRegression()),
                 ('Multinomial Naive Bayes', MultinomialNB()),
                 ('Ridge Regression', RidgeClassifier())]
voting_ensemble = VotingClassifier(base_learners)
classifiers = [*base_learners, ('Voting', voting_ensemble)]
# for features in n_features:
#     for n_grams in n_grams_range:
#         check_features_ngrams(features, n_grams, classifiers, colours)

performance_df = pd.read_csv('Data/Sentiment Analysis Performance Evaluation.csv')

logreg_unigram = performance_df[
    (performance_df['n_grams'] == 1) & 
    (performance_df['Classifier'] == 'Logistic Regression')][['Testing Accuracy']]
logreg_bigram = performance_df[
    (performance_df['n_grams'] == 2) & 
    (performance_df['Classifier'] == 'Logistic Regression')][['Testing Accuracy']]
logreg_trigram = performance_df[
    (performance_df['n_grams'] == 3) & 
    (performance_df['Classifier'] == 'Logistic Regression')][['Testing Accuracy']]
        
nb_unigram = performance_df[
    (performance_df['n_grams'] == 1) & 
    (performance_df['Classifier'] == 'Multinomial Naive Bayes')][
        ['Testing Accuracy']]
nb_bigram = performance_df[
    (performance_df['n_grams'] == 2) & 
    (performance_df['Classifier'] == 'Multinomial Naive Bayes')][
        ['Testing Accuracy']]
nb_trigram = performance_df[
    (performance_df['n_grams'] == 3) & 
    (performance_df['Classifier'] == 'Multinomial Naive Bayes')][
        ['Testing Accuracy']]
        
ridge_unigram = performance_df[
    (performance_df['n_grams'] == 1) & 
    (performance_df['Classifier'] == 'Ridge Regression')][['Testing Accuracy']]
ridge_bigram = performance_df[
    (performance_df['n_grams'] == 2) & 
    (performance_df['Classifier'] == 'Ridge Regression')][['Testing Accuracy']]
ridge_trigram = performance_df[
    (performance_df['n_grams'] == 3) & 
    (performance_df['Classifier'] == 'Ridge Regression')][['Testing Accuracy']]
        
voting_unigram = performance_df[(performance_df['n_grams'] == 1) & 
                                (performance_df['Classifier'] == 'Voting')][
                                    ['Testing Accuracy']]
voting_bigram = performance_df[(performance_df['n_grams'] == 2) & 
                               (performance_df['Classifier'] == 'Voting')][
                                   ['Testing Accuracy']]
voting_trigram = performance_df[(performance_df['n_grams'] == 3) & 
                                (performance_df['Classifier'] == 'Voting')][
                                    ['Testing Accuracy']]
                                    
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(n_features, logreg_unigram, label = 'Unigram')
axes[0, 0].plot(n_features, logreg_bigram, label = 'Bigram')
axes[0, 0].plot(n_features, logreg_trigram, label = 'Trigram')
axes[0, 0].set_title('Logistic Regression')
axes[0, 0].set_xscale('log')
axes[0, 0].set_ylabel('Accuracy')
# axes[0, 0].set_yticks([0.710, 0.715, 0.720, 0.725])
axes[0, 0].legend(loc = 'lower right')

axes[0, 1].plot(n_features, nb_unigram, label = 'Unigram')
axes[0, 1].plot(n_features, nb_bigram, label = 'Bigram')
axes[0, 1].plot(n_features, nb_trigram, label = 'Trigram')
axes[0, 1].set_title('Multinomial Naive Bayes')
axes[0, 1].set_xscale('log')
# axes[0, 1].set_yticks([0.700, 0.705, 0.710, 0.715])
axes[0, 1].legend(loc = 'lower right')

axes[1, 0].plot(n_features, ridge_unigram, label = 'Unigram')
axes[1, 0].plot(n_features, ridge_bigram, label = 'Bigram')
axes[1, 0].plot(n_features, ridge_trigram, label = 'Trigram')
axes[1, 0].set_title('Ridge Regressison')
axes[1, 0].set_xlabel('Features')
axes[1, 0].set_xscale('log')
axes[1, 0].set_ylabel('Accuracy')
# axes[1, 0].set_yticks([0.705, 0.710, 0.715])
axes[1, 0].legend(loc = 'lower right')

axes[1, 1].plot(n_features, voting_unigram, label = 'Unigram')
axes[1, 1].plot(n_features, voting_bigram, label = 'Bigram')
axes[1, 1].plot(n_features, voting_trigram, label = 'Trigram')
axes[1, 1].set_title('Voting')
axes[1, 1].set_xlabel('Features')
axes[1, 1].set_xscale('log')
# axes[1, 1].set_yticks([0.710, 0.715, 0.720])
axes[1, 1].legend(loc = 'lower right')

#%% Real-Time Tweet Classification
tf_idf_voting = TfidfVectorizer(max_features = 30000, ngram_range = (1, 2),
                                stop_words = 'english')
tf_idf_voting.fit(preprocessed_data['Tweet'])
tf_idf_transformed = tf_idf_voting.transform(preprocessed_data['Tweet'])
train_features = tf_idf_transformed[:10000]
train_target = preprocessed_data['Polarity'][:10000].values
voting_ensemble.fit(train_features, train_target)

class StreamClassifier(StreamListener):
    '''
    StreamListener is present in tweepy v3.10.0 and before. 
    For tweepy v4 onwards, use Stream instead (not recommended in this context)
    '''
    def __init__(self, classifier, vectorizer, time_limit = 60, api = None):
        super().__init__(api)
        self.start_time = time.time()
        self.time_limit = time_limit
        self.clf = classifier
        self.vec = vectorizer
        
    def on_data(self, data):
        '''
        Handles the incoming tweet
        '''
        if (time.time() - self.start_time) < self.time_limit:
            json_format = json.loads(data) # Create the json object
            tweet = json_format['text']
            features = self.vec.transform([tweet])
            predicted_sentiments = self.clf.predict(features)
            print(tweet, predicted_sentiments)
            real_time_streaming_df = pd.DataFrame(
                {'Tweet': tweet, 'Polarity': predicted_sentiments})
            if not os.path.isfile('Data/Real-Time Streaming Classification.csv'):
                real_time_streaming_df.to_csv(
                    'Data/Real-Time Streaming Classification.csv', index = False)
            else:
                real_time_streaming_df.to_csv(
                    'Data/Real-Time Streaming Classification.csv', mode = 'a', 
                    header = False, index = False)
            return True
        return False
    
    def on_error(self, status):
        '''
        Error handling -> Print the status
        '''
        print(status)
        
consumer_key = '<YOUR CONSUMER KEY>'
consumer_secret = '<YOUR CONSUMER SECRET>' 
access_token = '<YOUR ACCESS TOKEN>'
access_token_secret = '<YOUR ACCESS TOKEN SECRET>'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
classifier = StreamClassifier(classifier = voting_ensemble, 
                              vectorizer = tf_idf_voting, time_limit = 600)

# Listen to tweets within Singapore 
# Note: Keyword ([track parameter]) and location cannot be filtered together
stream = Stream(auth, classifier)
stream.filter(locations = [103.585763, 1.203748, 104.041009, 1.481474])

stop = time.perf_counter()
print('\u2550'*100)
duration = stop - start
minutes = divmod(duration, 60)[0]
seconds = divmod(duration, 60)[1]
print(colored('Execution Duration: {:.2f}s ({:.1f}mins, {:.2f}s)'.format(
    duration, minutes, seconds), 'red'))
