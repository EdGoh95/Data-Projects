#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 7: Online Learning Model Serving
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2],
              [4, 0], [4, 4], [4, 5], [0, 1],
              [2, 2], [3, 2], [5, 5], [1, -1]])

patients = np.array([[0, 0], [0, 1]])

kmeans = MiniBatchKMeans(n_clusters = 2, n_init = 3, random_state = 0, batch_size = 6).fit(patients)

def model_update(X):
    global kmeans
    kmeans.partial_fit(X)
    print('The new cluster centres are:')
    print(kmeans.cluster_centers_)

def predict(X):
    model_update(X)
    predictions = kmeans.predict(X)
    print('Predictions:', predictions)

if __name__ == '__main__':
    predict([[0, 0], [0, 1]])
    predict([[10, 10], [10, 15]])
    predict([[10, 11], [11, 15]])
    predict([[11, 10], [11, 14]])
    predict([[0, 0], [0, 1]])