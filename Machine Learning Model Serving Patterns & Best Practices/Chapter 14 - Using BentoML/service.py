#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 14: Using BentoML
"""
import bentoml
import numpy as np

model_runner = bentoml.sklearn.get('randforestregression').to_runner()
randforest_regression_service = bentoml.Service('RandForestRegressionService', runners = [model_runner])

@randforest_regression_service.api(input = bentoml.io.NumpyNdarray(), output = bentoml.io.NumpyNdarray(),
                                   route = '/bento-infer')
def predict(inp: np.ndarray) -> np.ndarray:
    print('Input:', inp)
    response = model_runner.run(inp)
    print('Response:', response)
    return response