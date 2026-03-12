#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 10: Ensemble Model Serving Pattern
"""
import pickle
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

X, Y = make_regression(n_features = 2, random_state = 0, shuffle = False, n_samples = 20)
randforest_model = RandomForestRegressor(max_depth = 2)
randforest_model.fit(X, Y)
print(randforest_model.predict([[0, 0]]))
pickle.dump(randforest_model, open('RandForest Regression Model.pb', 'wb'))

adaboost_model = AdaBoostRegressor(n_estimators = 5)
adaboost_model.fit(X, Y)
print(adaboost_model.predict([[0, 0]]))
pickle.dump(adaboost_model, open('AdaBoost Regression Model.pb', 'wb'))