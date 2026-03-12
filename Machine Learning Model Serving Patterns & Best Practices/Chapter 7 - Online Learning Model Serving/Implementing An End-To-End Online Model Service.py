#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 7: Online Learning Model Serving
"""
import numpy as np
import json
from sklearn.linear_model import SGDRegressor
from flask import *

X = [[1, 1, 1], [1, 1, 1], [1, 1, 1],
     [2, 2, 2], [2, 2, 2], [2, 2, 2]]
Y = [1, 1, 1, 2, 2, 2]

regression_model = SGDRegressor()
regression_model.fit(X, Y)
print('Initial Model - Coefficients: {}, Intercept: {}'.format(
    regression_model.coef_, regression_model.intercept_))

online_model_serving_app = Flask(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def model_update(Xnew, Ynew):
    print('Updating the model...')
    regression_model.partial_fit(Xnew, Ynew)
    print('After Update - Coefficients: {}, Intercept: {}'.format(
        regression_model.coef_, regression_model.intercept_))

@online_model_serving_app.route('/predict-online', methods = ['POST'])
def predict_online():
    X = json.loads(request.data)
    print('Input Data:', X)
    predictions = regression_model.predict(X)
    model_update(X, predictions)
    return json.dumps(predictions.tolist(), cls = NumpyEncoder)

if __name__ == '__main__':
    online_model_serving_app.run()