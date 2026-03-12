#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 13: Using Ray Serve
"""
import pandas as pd
from ray import remote, get
from ray.dag.input_node import InputNode
from sklearn.ensemble import RandomForestRegressor

@remote
def data_collection() -> pd.DataFrame:
    df = pd.DataFrame({'Feature 1': [1, 2, 3, 4, 5, None],
                       'Feature 2': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
                       'Y': [0, 0, 0, 1, 1, 1]})
    print('DataFrame in the data collection stage:')
    print(df)
    return df

@remote
def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    print('DataFrame after the data cleaning stage:')
    print(df)
    return df

@remote
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, dtype = int)
    print('DataFrame after the feature engineering stage:')
    print(df)
    return df

@remote
def train(df: pd.DataFrame) -> RandomForestRegressor:
    X = df[['Feature 1', 'Feature 2_C1']].values
    Y = df['Y']
    print('DataFrame is currently being used to train a RandomForest regression model')
    randforest_model = RandomForestRegressor(max_depth = 2, random_state = 0).fit(X, Y)
    return randforest_model

@remote
def predict(model: RandomForestRegressor, X) -> float:
    print('Obtaining prediction from the RandomForest regression model')
    print('Input:', X)
    return model.predict(X)

with InputNode() as X:
    data_collection = data_collection.bind()
    data_cleaning = data_cleaning.bind(data_collection)
    feature_engineering = feature_engineering.bind(data_cleaning)
    model_training = train.bind(feature_engineering)
    output = predict.bind(model_training, X)

print(get(output.execute([[1, 1]])))