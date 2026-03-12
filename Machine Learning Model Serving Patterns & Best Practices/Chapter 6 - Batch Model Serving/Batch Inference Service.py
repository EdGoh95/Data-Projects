#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 6: Batch Model Serving
"""
import pandas as pd
from flask import *

inference_app = Flask(__name__)

@inference_app.route('/predict_batch', methods = ['POST'])
def predict_with_params():
    predictions_df = pd.read_csv('Predictions.csv')
    predictions = []
    for index, row in predictions_df.iterrows():
        predictions.append((row['Product'], row['Score']))
    predictions.sort(key = lambda a: a[1], reverse = True)
    return jsonify(predictions)

inference_app.run()