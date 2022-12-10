#!/usr/bin/env python3
"""
Validating Your Code, Data And Models: Rethinking Unit Testing For Data Quality
"""
import pandas as pd

def test_upper():
    assert 'foo'.upper() == 'FOO'

df = pd.DataFrame(data = {'Data': [0, 1, 2, 3]})
def test_min():
    assert df.min().values[0] > 0
def test_max():
    assert df.max().values[0] < 100