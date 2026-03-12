#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 13: Using Ray Serve
"""
from ray import remote, get
from ray.dag.input_node import InputNode

@remote
def Step1(inp: str) -> str:
    print('I am in Step 1')
    return '{}_step1'.format(inp)

@remote
def Step2(inp: str) -> str:
    print('I am in Step 2')
    return '{}_step2'.format(inp)

@remote
def Step3(inp: str) -> str:
    print('I am in Step 3')
    return '{}_step3'.format(inp)

@remote
def Step4(inp: str) -> str:
    print('I am in Step 4')
    return '{}_step4'.format(inp)

@remote
class Model:
    def __init__(self):
        self.model = lambda x: '{}_predict'.format(x)

    def predict(self, inp: str) -> str:
        print('I am in the predict method')
        return self.model(inp)

with InputNode() as inp:
    step1 = Step1.bind(inp)
    step2 = Step2.bind(step1)
    step3 = Step3.bind(step2)
    step4 = Step4.bind(step3)
    model = Model.remote()
    output = model.predict.bind(step4)

print(get(output.execute('Hello!')))