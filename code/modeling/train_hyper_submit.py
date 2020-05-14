# Load required packages
from azureml.core import Workspace, Experiment
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.runconfig import MpiConfiguration
from azureml.train.dnn import PyTorch
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import (
    BayesianParameterSampling,
    HyperDriveConfig, PrimaryMetricGoal
)
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
import os


# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())


# Define Run Configuration
estimator = PyTorch(
    entry_script='train.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target='local',
    distributed_training=MpiConfiguration(),
    framework_version='1.4',
    use_gpu=False,
    conda_dependencies_file=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../',
        'conda_dependencies.yml'
    )
)


# Set parameters for search
param_sampling = BayesianParameterSampling({
    'learning_rate': uniform(0.05, 0.1),
    'num_epochs': choice(5, 10, 15),
    'batch_size': choice(150, 200),
    'hidden_size': choice(50, 100)
})


# Define multi-run configuration
hyperdrive_run_config = HyperDriveConfig(
    estimator=estimator,
    hyperparameter_sampling=param_sampling,
    policy=None,
    primary_metric_name='accuracy',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=80,
    max_concurrent_runs=None
)


# Define the ML experiment
experiment = Experiment(workspace, 'newsgroups_train_hypertune')


# Submit the experiment
hyperdrive_run = experiment.submit(hyperdrive_run_config)
hyperdrive_run.wait_for_completion()


# Select the best run from all submitted

# Log the best run's performance to the parent run

# Best set of parameters found
