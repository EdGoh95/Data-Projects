#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 5: Keyed Prediction
"""
import random
import asyncio

truths = [10, 11, 12, 13]
class Model:
    def __init__(self, name):
        self.name = name

    def predict(self, X, index):
        return truths[index:index+len(X)]

async def predict(modelName, data, model: Model, response):
    sleepTime = random.randint(0, 10)
    await asyncio.sleep(sleepTime)
    (x, i) = data
    print("{} prediction for {}:".format(modelName, x))
    y = model.predict(x, i)
    print("Response for {} - {}".format(x, y))
    await response.put(y)

async def server():
    X = [[1], [2], [3], [4]]
    responses = asyncio.Queue()
    model1 = Model("Model 1")
    model2 = Model("Model 2")
    model3 = Model("Model 3")
    model4 = Model("Model 4")
    await asyncio.gather(asyncio.create_task(predict("Model 1", (X[0], 0), model1, responses)),
                         asyncio.create_task(predict("Model 2", (X[1], 1), model2, responses)),
                         asyncio.create_task(predict("Model 3", (X[2], 2), model3, responses)),
                         asyncio.create_task(predict("Model 4", (X[3], 3), model4, responses)))
    print(responses)

if __name__ == "__main__":
    asyncio.run(server())