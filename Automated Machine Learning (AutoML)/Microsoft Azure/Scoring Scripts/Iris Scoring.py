import os
import joblib
import numpy as np
import pandas as pd
from azureml.core import Dataset, Datastore, Model, Run

iris_run = Run.get_context()

def main():
	# Initializing The Workspace & Dataset
	ws = iris_run.experiment.workspace
	datastore = Datastore.get_default(ws)
	iris_dataset = Dataset.get_by_name(ws, 'Iris (Simulated) Dataset For Scoring')
	iris_scoring_df = iris_dataset.to_pandas_dataframe()
	
	# Importing & Setting Up The Model
	iris_model = joblib.load(Model.get_model_path(model_name = 'Iris-MultiClass-Classification-AutoML', _workspace = ws))
	iris_predictions = pd.Series(iris_model.predict(iris_scoring_df))
	iris_scoring_df['Prediction'] = iris_predictions

	os.makedirs('Results', exist_ok = True)
	iris_scoring_df.to_csv('Results/Iris Predictions.csv', index = False, sep = ',')
	datastore.upload_files(files = ['Results/Iris Predictions.csv'], target_path = 'Results', overwrite = True)

	os.remove('Results/Iris Predictions.csv')
	os.rmdir('Results')

if __name__ == '__main__':
	main()