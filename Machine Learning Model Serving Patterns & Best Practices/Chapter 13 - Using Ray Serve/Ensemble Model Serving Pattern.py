#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 13: Using Ray Serve
"""
from ray.serve import deployment, run
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

@deployment
class RandForestModel:
    def __init__(self):
        X, Y = make_regression(n_features = 4, n_informative = 2, random_state = 0, shuffle = False)
        self.model = RandomForestRegressor(max_depth = 2, random_state = 0)
        self.model.fit(X, Y)

    def __call__(self, data):
        prediction = self.model.predict(data['X'])
        print('Prediction from the RandomForest model:', prediction)
        return prediction

@deployment
class AdaBoostModel:
    def __init__(self):
        X, Y = make_regression(n_features = 4, n_informative = 2, random_state = 0, shuffle = False)
        self.model = AdaBoostRegressor(random_state = 0, n_estimators = 100)
        self.model.fit(X, Y)

    def __call__(self, data):
        prediction = self.model.predict(data['X'])
        print('Prediction from the AdaBoost model:', prediction)
        return prediction

@deployment
class EnsembleModel:
    def __init__(self, ModelA_handle, ModelB_handle):
        self._ModelA_handle = ModelA_handle
        self._ModelB_handle = ModelB_handle

    async def __call__(self, request):
        data = await request.json()
        responseA = await self._ModelA_handle.remote(data)
        responseB = await self._ModelB_handle.remote(data)
        return (responseA + responseB)/2.0

ensemble_model = EnsembleModel.bind(RandForestModel.bind(), AdaBoostModel.bind())
run(ensemble_model)