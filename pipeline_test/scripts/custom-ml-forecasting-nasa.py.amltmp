import os
import shutil
import azureml.core
import pandas as pd
import numpy as np
import math
import joblib
import logging
import warnings
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from pandas import Grouper
from pandas import concat
from azureml.core.run import Run
# from azureml.widgets import RunDetails
from azureml.core import Dataset
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.train.estimator import Estimator
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from azureml.core import Environment
from azureml.core.runconfig import DockerConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# As part of the setup you have already created a <b>Workspace</b>. To run AutoML, you also need to create an <b>Experiment</b>. An Experiment corresponds to a prediction problem you are trying to solve, while a Run corresponds to a specific approach to the problem.

ws = Workspace.get(name="mlprojectA",
       subscription_id='e6ec79e7-c7e5-4312-85d0-75e8285c09dd',
       resource_group='PartA-projectA')

# choose a name for the run history container in the workspace
experiment_name = "nasa-battery-forecast"

experiment = Experiment(ws, experiment_name)

project_folder = './train-on-amlcompute'
os.makedirs(project_folder, exist_ok=True)
shutil.copy('train.py', project_folder)

myenv = Environment("myenv")
myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn', 'packaging'])

# Enable Docker
docker_config = DockerConfiguration(use_docker=True)

# Choose a name for your CPU cluster
cpu_cluster_name = "nasa-cluster-customml"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_D32s_v3',
                                                           max_nodes=1)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)

from azureml.core import ScriptRunConfig

src = ScriptRunConfig(source_directory=project_folder, 
                      script='train.py', 
                      compute_target=cpu_cluster, 
                      environment=myenv,
                      docker_runtime_config=docker_config)
run = experiment.submit(config=src)

# Change run name
submitter = run.get_details()['submittedBy'].split('(')[0]
runType   = run.get_details()['properties']['runType']
best_run, fitted_model = run.get_output()
bestrun = best_run.get_properties()
runalgorithm = bestrun['run_algorithm']
run.display_name = submitter + '/' + str(cnt4nm) + '/' + runType + '/' + runalgorithm

# Shows output of the run on stdout.
run.wait_for_completion(show_output=True)

run.get_metrics()
