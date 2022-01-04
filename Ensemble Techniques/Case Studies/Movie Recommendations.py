#!/usr/bin/env python3
"""
Hands-On Ensemble Learning with Python (Packt Publishing) Chapter 12:
Recommending Movies with Keras
"""
import time
import numpy as np
np.random.seed(123456)
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model

start = time.perf_counter()
#%% Load The Dataset
movie_ratings_df = pd.read_csv('Data/MovieLens/ratings.csv')
display(movie_ratings_df.head())
movie_ratings_df.drop('timestamp', axis = 1, inplace = True)

users = movie_ratings_df['userId'].unique()
n_users = len(users)
movies = movie_ratings_df['movieId'].unique()
n_movies = len(movies)
# Create maps for users and movies from old to new indices
moviemap, usermap = {}, {}
for i in range(n_movies):
    moviemap[movies[i]] = i
for j in range(n_users):
    usermap[users[j]] = j

movie_ratings_df['movieId'] = movie_ratings_df['movieId'].apply(lambda x: moviemap[x])
movie_ratings_df['userId'] = movie_ratings_df['userId'].apply(lambda x: usermap[x])
movie_ratings_shuffled_df = movie_ratings_df.sample(frac = 1).reset_index(drop = True)
training, testing = train_test_split(movie_ratings_shuffled_df, test_size = 0.2)
#%% Data Exploration and Analysis
movie_ratings_df['rating'].plot.hist(title = "Rating Distribution")
plt.xlabel('Rating')

#%% Creating The Neural Networks
#### Dot Model (Single Perceptron)
np.random.seed(123456)
n_features = 5
# Transforms a movie input into a 1D tensor via the Embedding layer
movie_input = Input(shape = [1], name = 'Movie')
movie_embedding = Embedding(n_movies, n_features, name = 'Movie_Embedding')(movie_input)
# Flatten the output of the Embedding layer into a 1D tensor
movie_flattened = Flatten(name = 'Movie_Flattened')(movie_embedding)

# Do the same transformation to the user input
user_input = Input(shape = [1], name = 'User')
user_embedding = Embedding(n_users, n_features, name = 'User_Embedding')(user_input)
user_flattened = Flatten(name = 'User_Flattened')(user_embedding)

# Obtain the dot product of both movie and user parts
dot_product = Dot(name = 'Dot_Product', axes = 1)([movie_flattened, user_flattened])

# Instantiate and compile the model
dot_model = Model([user_input, movie_input], dot_product)
dot_model.compile('adam', 'mean_squared_error')
print(colored('\u2500'*44 + ' Dot Model ' + '\u2500'*44, 'blue', attrs = ['bold']))
dot_model.summary() # Output the summary of the model

# Train and evaluate the model
dot_model.fit([training['userId'], training['movieId']], training['rating'], epochs = 20,
              verbose = 1)
print('Mean Squared Error (MSE): {:.3f}'.format(mean_squared_error(
    testing['rating'], dot_model.predict([testing['userId'], testing['movieId']]))))

#### Dense Model
# Concatenate the movie and user Embedding layers
np.random.seed(123456)
concatenated = Concatenate()([movie_flattened, user_flattened])
# Feed the concatenated Embedding layers into the dense part of the neural network
dense1 = Dense(128)(concatenated)
dense2 = Dense(32)(dense1)
output = Dense(1)(dense2)

# Instantiate and compile the model
dense_model = Model([user_input, movie_input], output)
dense_model.compile('adam', 'mean_squared_error')
print(colored('\u2500'*43 + ' Dense Model ' + '\u2500'*43, 'green', attrs = ['bold']))
dense_model.summary() # Output the summary of the model

# Train and evaluate the model
dense_model.fit([training['userId'], training['movieId']], training['rating'],
                epochs = 225, verbose = 1)
print('Mean Squared Error (MSE): {:.3f}'.format(mean_squared_error(
    testing['rating'], dense_model.predict([testing['userId'], testing['movieId']]))))

#### Stacking Ensemble
def create_model(n_features = 5, train_model = True, load_weights = False):
    np.random.seed(123456)
    # Transforms a movie input into a 1D tensor via the Embedding layer
    movie_input = Input(shape = [1], name = 'Movie')
    movie_embedding = Embedding(
        n_movies, n_features, name = 'Movie_Embedding')(movie_input)
    # Flatten the output of the Embedding layer into a 1D tensor
    movie_flattened = Flatten(name = 'Movie_Flattened')(movie_embedding)

    # Do the same transformation to the user input
    user_input = Input(shape = [1], name = 'User')
    user_embedding = Embedding(n_users, n_features, name = 'User_Embedding')(user_input)
    user_flattened = Flatten(name = 'User_Flattened')(user_embedding)

    # Concatenate the movie and user Embedding layers
    concatenated = Concatenate()([movie_flattened, user_flattened])
    # Feed the concatenated Embedding layers into the dense part of the neural network
    dense1 = Dense(128)(concatenated)
    dense2 = Dense(32)(dense1)
    output = Dense(1)(dense2)
    # Instantiate and compile the model
    model = Model([user_input, movie_input], output)
    model.compile('adam', 'mean_squared_error')
    model.fit([training['userId'], training['movieId']], training['rating'], epochs = 10,
              verbose = 1)
    # Generate the model predictions
    predictions = model.predict([testing['userId'], testing['movieId']])
    return model, predictions

# Instantiate the base learners and generate the predictions for each base learner
print(colored('\u2500'*45 + ' Stacking ' + '\u2500'*45, 'yellow', attrs = ['bold']))
model_5, predictions_5 = create_model(n_features = 5)
model_10, predictions_10 = create_model(n_features = 10)
model_15, predictions_15 = create_model(n_features = 15)
# Combining the predictions into a single array
predictions = np.stack([predictions_5, predictions_10, predictions_15],
                       axis = -1).reshape(-1, 3)
print(colored('Base Learners', 'magenta', attrs = ['bold']))
print('With 5 features - Mean Squared Error (MSE): {:.4f}'.format(mean_squared_error(
    testing['rating'][-1000:], predictions_5[-1000:])))
print('With 10 features - Mean Squared Error (MSE): {:.4f}'.format(mean_squared_error(
    testing['rating'][-1000:], predictions_10[-1000:])))
print('With 15 features - Mean Squared Error (MSE): {:.4f}'.format(mean_squared_error(
    testing['rating'][-1000:], predictions_15[-1000:])))

meta_learner = BayesianRidge()
meta_learner.fit(predictions[:-1000], testing['rating'][:-1000])
print(colored('\nEnsemble', 'cyan', attrs = ['bold']))
print('Mean Squared Error (MSE): {:.4f}'.format(mean_squared_error(
    testing['rating'][-1000:], meta_learner.predict(predictions[-1000:]))))

stop = time.perf_counter()
print('\u2550'*100)
duration = stop - start
hours = divmod(divmod(duration, 60), 60)[0][0]
minutes = divmod(divmod(duration, 60), 60)[1][0]
seconds = divmod(divmod(duration, 60), 60)[1][1]
print(colored('Execution Duration: {:.2f}s ({:.1f}hrs, {:.1f}mins, {:.2f}s)'.format(
    duration, hours, minutes, seconds), 'red'))