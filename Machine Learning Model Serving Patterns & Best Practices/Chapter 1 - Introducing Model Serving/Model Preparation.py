#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 1: Introducing Model Serving
"""
import bentoml
from sklearn import svm, datasets

iris_dataset = datasets.load_iris()
iris_features, iris_species = iris_dataset.data, iris_dataset.target

SVC_classifier = svm.SVC(gamma = 'scale')
SVC_classifier.fit(iris_features, iris_species)
saved_iris_model = bentoml.sklearn.save_model('iris-svc-model', SVC_classifier)
print('Model saved as', saved_iris_model)