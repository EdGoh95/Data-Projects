#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 16: Integrating ML Systems in Ecosystems
"""
import pandas as pd
from flask import *
from joblib import load

ML_app = Flask(__name__)

@ML_app.route('/predict/<LOC>/<MCC>')
def predict(LOC, MCC):
    return {'Defect': generate_predictions(LOC, MCC)}

@ML_app.route('/')
def welcome():
    return """Welcome to the predictor! You need to send a GET request with the following parameters -
    LOC: Lines of Codes, MCC: McCabe Complexity"""

def generate_predictions(LOC, MCC):
    dt_model = load("dt.joblib")
    features = pd.DataFrame({'LOC': LOC, 'MCC': MCC}, index = [0])
    predictions = dt_model.predict(features)
    return int(predictions[0])

if __name__ == '__main__':
    ML_app.run( host = '0.0.0.0', port = 5001, threaded = True, debug = True)