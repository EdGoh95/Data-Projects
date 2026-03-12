#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 13: Using Ray Serve
"""
from ray import serve

@serve.deployment
class ModelA:
    def __init__(self):
        self.model = lambda x: x + 5

    def __call__(self, data):
        return self.model(data['X'])

@serve.deployment
class ModelB:
    def __init__(self):
        self.model = lambda x: x * 2

    def __call__(self, data):
        return self.model(data['X'])

@serve.deployment
class Driver:
    def __init__(self, ModelA_handle, ModelB_handle):
        self._ModelA_handle = ModelA_handle
        self._ModelB_handle = ModelB_handle

    async def __call__(self, request):
        data = await request.json()
        responseA = await self._ModelA_handle.remote(data)
        responseB = await self._ModelB_handle.remote(data)
        return responseA + responseB

driver = Driver.bind(ModelA.bind(), ModelB.bind())
serve.run(driver)