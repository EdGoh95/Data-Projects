#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 14: Using BentoML
"""
import bentoml
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

X, Y = make_regression(n_features = 4, n_informative = 2, random_state = 0, shuffle = False)
randforest = RandomForestRegressor(max_depth = 3, random_state = 0).fit(X, Y)
randforest_model = bentoml.sklearn.save_model(name = 'RandForest', model = randforest)
adaboost = AdaBoostRegressor(random_state = 0).fit(X, Y)
adaboost_model = bentoml.sklearn.save_model(name = 'AdaBoost', model = adaboost)

randforest_runner = bentoml.sklearn.get('randforest').to_runner()
adaboost_runner = bentoml.sklearn.get('adaboost').to_runner()
regression_service = bentoml.Service('regression_service',
                                     runners = [randforest_runner, adaboost_runner])

@regression_service.api(input = bentoml.io.NumpyNdarray(), output = bentoml.io.NumpyNdarray(),
                        route = '/infer')
def predict(inp: np.ndarray) -> np.ndarray:
    print('Input:', inp)
    randforest_response = randforest_runner.run(inp)
    print('Prediction from the RandForest model:', randforest_response)
    adaboost_response = adaboost_runner.run(inp)
    print('Prediction from the AdaBoost model:', adaboost_response)
    ensemble = (randforest_response + adaboost_response)/2
    print('Average prediction from both models:', ensemble)
    return ensemble