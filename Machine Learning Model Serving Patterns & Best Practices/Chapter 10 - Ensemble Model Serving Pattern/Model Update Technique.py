#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 10: Ensemble Model Serving Pattern
"""
def load_model(filename):
    print('Loading model', filename)

def current_model_prediction(X):
    model = load_model('../Source Code From GitHub/Chapter 10/model_update/model_current/model.txt')
    print("Current model is predicting for", X)
    return "Prediction from current model"

def new_model_prediction(X):
    model = load_model('../Source code From GitHub/Chapter 10/model_update/model_new/model.txt')
    print("New model is predicting for", X)
    return "Prediction from new model"

def predict(evaluation_period, X):
    if evaluation_period == True:
        current_prediction = current_model_prediction(X)
        new_prediction = new_model_prediction(X)
        file = open('evaluation_data.csv', 'a')
        file.write('{}, {}'.format(current_prediction, new_prediction))
        return current_prediction
    else:
        return new_model_prediction(X)