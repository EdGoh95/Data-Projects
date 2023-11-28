import os
import argparse
import numpy as np
import pandas as pd
from azureml.core import Datastore, Run

iris_parallel_run = Run.get_context()
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_folder', type = str)
args = parser.parse_args()

def main():
	# Transfer The Predictions From An Intermediate Pipeline Data Location To Its Final Destination
	iris_parallel_prediction_df = pd.read_csv(os.path.join(args.input_data_folder, 'parallel_run_step.txt'), delimiter = ' ', header = None)
	iris_parallel_prediction_df.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Prediction']

	# Initializing The Workspace
	ws = iris_parallel_run.experiment.workspace
	datastore = Datastore.get_default(ws)

	os.makedirs('Results', exist_ok = True)
	iris_parallel_prediction_df.to_csv('Results/Iris Parallel Predictions.csv', index = False, sep = ',')
	datastore.upload_files(files = ['Results/Iris Parallel Predictions.csv'], target_path = 'Results', overwrite = True)

	os.remove('Results/Iris Parallel Predictions.csv')
	os.rmdir('Results')

if __name__ == '__main__':
	main()