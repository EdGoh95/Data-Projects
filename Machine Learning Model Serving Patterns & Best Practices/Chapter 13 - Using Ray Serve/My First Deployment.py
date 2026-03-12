#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 13: Using Ray Serve
"""
from ray import serve

@serve.deployment
class MyFirstRayServeDeployment:
    def __init__(self, message):
        self.message = message

    def __call__(self):
        return self.message

my_first_deployment = MyFirstRayServeDeployment.bind('Hello World!')