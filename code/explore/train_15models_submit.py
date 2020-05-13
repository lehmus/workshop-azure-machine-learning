'''
Training submitter

Facilitates (remote) training execution through the Azure ML service.
'''

# impot required packages
import os
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication

# load Azure ML workspace
workspace = Workspace.from_config(auth=AzureCliAuthentication())

# Define Run Configuration
est = Estimator(
    entry_script='train_15models.py',
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    compute_target='local',
    user_managed=True,
    use_docker=False
)

# Define the ML experiment
experiment = Experiment(workspace, 'newsgroups_train_15models')

# Submit experiment run, if compute is idle, this may take some time
run = experiment.submit(est)

# wait for run completion of the run, while showing the logs
run.wait_for_completion(show_output=True)

# Select best model from run
