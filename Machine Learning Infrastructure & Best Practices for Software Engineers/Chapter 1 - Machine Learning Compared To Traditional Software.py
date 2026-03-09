#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 1: Machine Learning Compared To Traditional Software
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#%% Traditional Fibonacci Sequence Algorithm
def fibRec(n):
    """
    A recursive function to calculate the Fibonacci number
    """
    if n < 2:
        return n
    else:
        return fibRec(n - 1) + fibRec(n - 2)

#%% Predicting The Fibonacci Number Using Linear Regression
#### Data Preparation
# First 2 columns are the numbers in the sequence and the third column is the result of the sequence
training_df = pd.DataFrame([[1, 1, 2],
                            [2, 1, 3],
                            [3, 2, 5],
                            [5, 3, 8],
                            [8, 5, 13]], columns = ['First Number', 'Second Number', 'Result'])
# List containing the first 2 numbers of the Fibonacci Sequence
lstSequence = [0, 1]

#### Model Training & Inference
LR_model = LinearRegression()
LR_model.fit(training_df[['First Number', 'Second Number']], training_df['Result'])
print('Linear Regression Model Score: {:.2f}'.format(
    LR_model.score(training_df[['First Number', 'Second Number']], training_df['Result'])))

for k in range(23):
    # Stores the 2 numbers in the lstSequence list as an array and makes the prediction
    # The result returned from the model is a float, hence it is necessary to convert it to an integer
    intFibonacci = int(np.round(LR_model.predict(np.array([[lstSequence[k], lstSequence[k+1]]]))))
    lstSequence.append(intFibonacci)
print(lstSequence)