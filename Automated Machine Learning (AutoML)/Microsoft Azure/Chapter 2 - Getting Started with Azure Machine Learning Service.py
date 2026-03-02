#!/usr/bin/env python3
"""
Automated Machine Learning With Microsoft Azure (Packt Publishing)
Chapter 2: Getting Started with Azure Machine Learning Service
"""
from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.opendatasets import Diabetes
from azureml.core import Dataset

#%% Create A Compute Instance
workspace = Workspace.from_config(path = 'config.json')
compute_name = 'automl-cluster'
compute_configuration = AmlCompute.provisioning_configuration(
    vm_size = 'Standard_DS3_v2', vm_priority = 'lowpriority',min_nodes = 0, max_nodes = 4,
    idle_seconds_before_scaledown = 1200)
compute_target = ComputeTarget.create(workspace, compute_name, compute_configuration)
compute_target.wait_for_completion(show_output = True)

#%% Create A Dataset
diabetes_tabular = Diabetes.get_tabular_dataset()
diabetes = diabetes_tabular.register(workspace = workspace, name = 'Diabetes Sample')