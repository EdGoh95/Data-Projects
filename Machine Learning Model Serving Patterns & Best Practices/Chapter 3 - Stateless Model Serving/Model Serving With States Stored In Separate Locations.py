#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 3: Stateless Model Serving
"""
import pickle
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from flask import *

X = [[8, 4, 2], [5, 1, 3], [5, 1, 3]]
Y = [2, 1, 0]

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X, Y)
# with open('Decision Tree Model.pkl', 'wb') as model_file:
#     pickle.dump(decision_tree_model, model_file)

model_params = decision_tree_model.__dict__
with open('State Store/Decision Tree Parameters.pkl', 'wb') as param_file:
    pickle.dump(model_params, param_file)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

model_serving_app = Flask(__name__)
@model_serving_app.route('/predict-using-params', methods = ['POST'])
def predict_params_loaded():
    with open('Decision Tree Model.pkl', 'rb') as file:
        model: DecisionTreeClassifier = pickle.load(file)
        params = pickle.load(open('State Store/Decision Tree Parameters.pkl', 'rb'))
        model.__dict__ = params
        features = json.loads(request.data)
        response = model.predict(features)
        return json.dumps(int(response), cls = NumpyEncoder)

@model_serving_app.route('/predict-using-full-model-loaded', methods = ['POST'])
def predict_full_model_loaded():
    with open('Decision Tree Model.pkl', 'rb') as file:
       model: DecisionTreeClassifier = pickle.load(file)
       features = json.loads(request.data)
       if len(features) == 0 or len(features[0]) != 3:
           abort(400, """The requested dimension does not match the expected dimension.
                 The input length of {} is {} instead of 3 which the model expects.""".format(
                 features[0], len(features[0])))
       response = model.predict(features)
       return json.dumps(int(response), cls = NumpyEncoder)

model_serving_app.run()