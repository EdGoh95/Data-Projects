#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 10: Ensemble Model Serving Pattern
"""
import pickle
import json
from flask import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

randforest_model: RandomForestRegressor = pickle.load(open('RandForest Regression Model.pb', 'rb'))
adaboost_model: AdaBoostRegressor = pickle.load(open('AdaBoost Regression Model.pb', 'rb'))

ensemble_model_serving_app = Flask(__name__)

def randforest_prediction(X):
    response = randforest_model.predict(X)
    print('Response from the RandomForest model:', response)
    return response

def adaboost_prediction(X):
    response = adaboost_model.predict(X)
    print('Response from the AdaBoost model:', response)
    return response

@ensemble_model_serving_app.route('/predict-using-ensemble-model', methods = ['POST'])
def ensemble_prediction():
    X = json.loads(request.data)
    response1 = randforest_prediction(X)
    response2 = adaboost_prediction(X)
    combined_response = (response1 + response2)/2
    return 'Final Prediction: {}'.format(combined_response)

ensemble_model_serving_app.run()