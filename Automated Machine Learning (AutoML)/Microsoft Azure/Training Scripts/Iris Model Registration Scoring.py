import argparse
from azureml.core import Run, Model, Dataset

iris_retraining_run = Run.get_context()

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', dest = 'model_name')
parser.add_argument('--model_path', dest = 'model_path')
parser.add_argument('--dataset_name', dest = 'dataset_name')
args = parser.parse_args()

def main():
	# Initializing The Workspace & Dataset
	ws = iris_retraining_run.experiment.workspace
	iris_dataset = [(Dataset.Scenario.TRAINING, Dataset.get_by_name(ws, args.dataset_name))]
	iris_model = Model.register(workspace = ws, model_path = args.model_path, model_name = args.model_name, datasets = iris_dataset)

if __name__ == '__main__':
	main()