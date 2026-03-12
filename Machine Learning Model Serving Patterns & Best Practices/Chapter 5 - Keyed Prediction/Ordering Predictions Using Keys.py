#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 5: Keyed Prediction
"""
import random

truths = [10, 11, 12, 13]
def jumbled_predictions_without_keys(X):
    response = []
    for i in range(len(X)):
        response.append(truths[i])
    random.shuffle(response)
    return response

def jumbled_predictions_with_keys(X):
    response = []
    for j in range(len(X)):
        (k, v) = X[j]
        response.append((k, truths[j]))
    random.shuffle(response)
    return response

if __name__ == '__main__':
    X1 = [[1], [2], [3], [4]]
    Y1 = jumbled_predictions_without_keys(X1)
    print("Predictions without keys:", Y1)
    X2 = [(0, 1), (1, 2), (2, 3), (3, 4)]
    Y2 = jumbled_predictions_with_keys(X2)
    print("Predictions with keys:", Y2)
    Y2.sort(key = lambda predictions: predictions[0])
    print("Restoring the order of predictions using keys:", Y2)