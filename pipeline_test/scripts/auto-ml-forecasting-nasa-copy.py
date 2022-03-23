import os
import azureml.core
import pandas as pd
import numpy as np
import logging
import warnings
import pandas as pd
from pandas import DataFrame
from pandas import Grouper
from pandas import concat
from datetime import datetime
from helper import split_full_for_forecasting
from azureml.core import Datastore
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

# As part of the setup you have already created a <b>Workspace</b>. To run AutoML, you also need to create an <b>Experiment</b>. An Experiment corresponds to a prediction problem you are trying to solve, while a Run corresponds to a specific approach to the problem.
print("starting automl.......................................................")

ws = Workspace.get(name="mlprojectA",
       subscription_id='e6ec79e7-c7e5-4312-85d0-75e8285c09dd',
       resource_group='PartA-projectA')

# choose a name for the run history container in the workspace
experiment_name = "nasa-battery-forecast"

experiment = Experiment(ws, experiment_name)

# Choose a name for your CPU cluster
cpu_cluster_name = "nasa-cluster-automl"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cluster, use it.")
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_D32s_v3", max_nodes=4
    )
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)


# ## Data

target_column_name = "Current_measured"
time_column_name = "actual_time"
time_series_id_column_names = []
freq = "S"  # Second data

# dataset = ws.datasets['nanadataset1']

dataset = Dataset.get_by_id(ws, "767b0781-3d98-4c76-832a-66a6fb969906")
df = dataset.to_pandas_dataframe()
cnt4nm = df.count(axis=1)[-1]
train, valid = split_full_for_forecasting(df, time_column_name)
train.to_csv("train.csv")
valid.to_csv("valid.csv")

datastore = ws.get_default_datastore()

train_dataset = Dataset.Tabular.from_delimited_files(
    path=[(datastore, "nasa-dataset/tabular/train.csv")]
    
)
valid_dataset = Dataset.Tabular.from_delimited_files(
    path=[(datastore, "nasa-dataset/tabular/valid.csv")]
)

# ### Setting forecaster maximum horizon 

forecast_horizon = 3600

# ## Train
# 
# Instantiate a AutoMLConfig object. This defines the settings and data used to run the experiment.

forecasting_parameters = ForecastingParameters(
    time_column_name=time_column_name,
    forecast_horizon=forecast_horizon,
    freq="S",  # Set the forecast frequency to be monthly (start of the month)
)

# We will disable the enable_early_stopping flag to ensure the DNN model is recommended for demonstration purpose.
automl_config = AutoMLConfig(
    task="forecasting",
    primary_metric="normalized_root_mean_squared_error",
    experiment_timeout_hours=12,
    training_data=train_dataset,
    label_column_name=target_column_name,
    validation_data=valid_dataset,
    verbosity=logging.INFO,
    compute_target=compute_target,
    max_concurrent_iterations=4,
    max_cores_per_iteration=-1,
    enable_dnn=True,
    enable_early_stopping=True,
    forecasting_parameters=forecasting_parameters,
)

# We will now run the experiment, starting with 10 iterations of model search. The experiment can be continued for more iterations if more accurate results are required. Validation errors and current status will be shown when setting `show_output=True` and the execution will be synchronous.

remote_run = experiment.submit(automl_config, show_output=True)

submitter = remote_run.get_details()['submittedBy'].split('(')[0]
runType   = remote_run.get_details()['properties']['runType']
best_run, fitted_model = remote_run.get_output()
bestrun = best_run.get_properties()
runalgorithm = bestrun['run_algorithm']
remote_run.display_name = submitter + '/' + str(cnt4nm) + '/' + runType + '/' + runalgorithm
