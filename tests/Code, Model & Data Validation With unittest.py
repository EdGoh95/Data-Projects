#!/usr/bin/env python3
"""
Validating Your Code, Data And Models: Rethinking Unit Testing For Data Quality
"""
import unittest
import pandas as pd

class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

class TestDataFrameStats(unittest.TestCase):
    def setup(self):
        # Initialize and load df
        self.df = pd.DataFrame(data = {'Data': [0, 1, 2, 3]})
    def test_min(self):
        self.assertGreatEqual(self.df.min().values[0], 0)
    def test_max(self):
        self.assertLessEqual(self.df.max().values[0], 100)