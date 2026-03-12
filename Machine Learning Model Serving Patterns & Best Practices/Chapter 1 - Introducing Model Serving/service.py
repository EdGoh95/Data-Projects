#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 1: Introducing Model Serving
"""
import bentoml
import numpy as np

iris_model_runner = bentoml.sklearn.get('iris-svc-model:latest').to_runner()
iris_classifier = bentoml.Service('iris-classifier', runners = [iris_model_runner])

@bentoml.service(resources = {'cpu': '10', 'memory': '2Gi'})
class Iris_Classification:
    @iris_classifier.api(input = bentoml.io.NumpyNdarray(), output = bentoml.io.NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        return iris_model_runner.predict.run(input_series)