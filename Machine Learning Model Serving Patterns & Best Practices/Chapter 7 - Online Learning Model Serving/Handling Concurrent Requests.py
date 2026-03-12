#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 7: Online Learning Model Serving
"""
import queue

Q = queue.Queue()
class Request():
    def __init__(self, ID, data):
        self.ID = ID
        self.data = data

def model_update(request):
    Q.put(request)
    if Q.qsize() >= 3:
        print('Model preparing to update now. Please wait...')
        while Q.qsize() > 0:
            request = Q.get()
        print('Queue size after update:', Q.qsize())
    else:
        print('Model not being updated yet')

request1 = Request('ID1', [[1, 1]])
model_update(request1)
request2 = Request('ID2', [[2, 2]])
model_update(request2)
request3 = Request('ID3', [[3, 3]])
model_update(request3)