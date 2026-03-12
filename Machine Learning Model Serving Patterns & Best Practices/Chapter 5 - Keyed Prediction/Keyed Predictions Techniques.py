#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 5: Keyed Prediction
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from termcolor import colored

features = ['Bedrooms', 'Bathrooms']

def train(X: pd.DataFrame, Y: pd.Series):
    X = X.loc[:, features]
    LR_model = LinearRegression()
    LR_model.fit(X.values, Y)
    return LR_model

def predict_single(model: LinearRegression, X, key):
    prediction = model.predict(X)
    return pd.DataFrame({'key': key, 'prediction': prediction})

def predict_batch(model: LinearRegression, X: pd.DataFrame):
    X = X.sample(frac = 1)
    keys = X['key']
    response = pd.DataFrame()
    for index, row in X.iterrows():
        predictions = predict_single(model, [row[features]], keys[index])
        response = pd.concat([response, predictions], ignore_index = True)
    return response.sort_values(by = 'key', ignore_index = True)

if __name__ == '__main__':
    X = pd.DataFrame({'Bedrooms': [5, 5, 4, 4, 3], 'Bathrooms': [4, 3, 3, 2, 2]})
    Y = pd.Series([500, 450, 400, 350, 300])
    model = train(X, Y)
    X['key'] = pd.Series(['1', '2', '3', '4', '5'])
    response = predict_batch(model, X)
    print(colored('Predictions:', 'cyan', attrs = ['bold']))
    print(response)