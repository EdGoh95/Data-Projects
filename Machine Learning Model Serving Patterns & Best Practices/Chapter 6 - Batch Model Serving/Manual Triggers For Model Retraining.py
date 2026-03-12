#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 6: Batch Model Serving
"""
import numpy as np
from termcolor import colored

Users = np.array(['A', 'B', 'C', 'D', 'E'])
Week1 = np.array([5, 5, 5, 5, 5])
Week2 = np.array([5, 4, 4, 5, 5])
Week3 = np.array([4, 4, 4, 4, 4])
Week4 = np.array([3, 3, 4, 4, 4])
Week5 = np.array([3, 3, 3, 3, 3])

def three_or_less(rating):
    mappedArray = rating <= 3
    return mappedArray.any()

print(colored("Checking any week with the first appearance of rating <= 3", 'green', attrs = ['bold']))
for index, week in enumerate([Week1, Week2, Week3, Week4, Week5]):
    print('Week {}: {}'.format(index+1, three_or_less(week)))

def check_average(rating):
    return np.mean(rating) <= 4.0

print(colored("Checking which weeks whose average rating <= 4.0", 'blue', attrs = ['bold']))
for index, week in enumerate([Week1, Week2, Week3, Week4, Week5]):
    print('Week {}: {}'.format(index+1, check_average(week)))

def check_median(rating):
    return np.median(rating) <= 3.0

print(colored("Checking which weeks whose median rating <= 3.0", 'red', attrs = ['bold']))
for index, week in enumerate([Week1, Week2, Week3, Week4, Week5]):
    print('Week {}: {}'.format(index+1, check_median(week)))