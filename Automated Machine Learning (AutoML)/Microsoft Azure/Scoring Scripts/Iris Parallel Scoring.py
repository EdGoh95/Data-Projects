import os
import joblib
import argparse
import numpy as np
import pandas as pd
from azureml.core import Model, Run

iris_parallel_run = Run.get_context()
ws = iris_parallel_run.experiment.workspace

def init():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', dest = 'model_name', required = True)
	args, unknown_args = parser.parse_known_args()
	global iris_model 
	iris_model = joblib.load(Model.get_model_path(model_name = 'Iris-MultiClass-Classification-AutoML', _workspace = ws))

def run(input_data):
	iris_predictions = pd.Series(iris_model.predict(input_data))
	input_data['Prediction'] = iris_predictions
	print('Data has been written into parallel_run_step.txt')
	return input_data
