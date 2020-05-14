'''
Training submitter for PyTorch neural network with HyperDrive hyperparameter tuning.
'''

import argparse
from azureml.core import Workspace, Experiment, Dataset
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.runconfig import MpiConfiguration
from azureml.train.dnn import PyTorch
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import (
    BayesianParameterSampling, HyperDriveConfig, PrimaryMetricGoal
)
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
import os
import sys


def get_arguments(args, arg_defs):
    '''
    Read and parse program arguments.
    '''

    parser = argparse.ArgumentParser()
    for arg_name in arg_defs.keys():
        short_flag = '-' + arg_defs[arg_name]['short_flag']
        long_flag = '--' + arg_name
        default_value = arg_defs[arg_name]['default_value']
        arg_type = arg_defs[arg_name]['type']
        description = arg_defs[arg_name]['description']
        parser.add_argument(
            short_flag, long_flag, dest=arg_name, type=arg_type,
            default=default_value, help=description
        )
    args = parser.parse_args(args[1:])
    return args


def main(argv):

    # Define and read program input arguments
    program_args = {
        'script_file': {
            'short_flag': 's',
            'type': str,
            'default_value': 'train.py',
            'description': 'Script file path on the host system.'
        },
        'cluster_name': {
            'short_flag': 'c',
            'type': str,
            'default_value': 'local',
            'description': 'Azure ML Compute cluster name.'
        },
        'experiment_name': {
            'short_flag': 'e',
            'type': str,
            'default_value': 'experiment',
            'description': 'Azure ML Experiment name.'
        },
        'hyperparam_runs_max': {
            'short_flag': 'r',
            'type': int,
            'default_value': 80,
            'description': 'Maximum number of runs for hyperparameter tuning.'
        },
        'hyperparam_concur_max': {
            'short_flag': 'm',
            'type': int,
            'default_value': 20,
            'description': (
                'Maximum number of concurrent runs for Bayesian sampling in hyperparameter tuning.'
                ' Use None to allow unlimited concurrent runs.'
            )
        },
        'log_level': {
            'short_flag': 'l',
            'type': str,
            'default_value': 'warning',
            'description': 'Logging output level.'
        }
    }
    args = get_arguments(argv, program_args)

    # PyTorch library version
    pytorch_version = '1.4'

    # Check if GPU computing is available
    is_gpucluster = ('gpu' in args.cluster_name)

    # Load Azure ML Workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())

    # Load Azure ML Datasets
    dataset_train = Dataset.get_by_name(workspace, name='newsgroups_subset_train')
    dataset_test = Dataset.get_by_name(workspace, name='newsgroups_subset_test')

    # Define Run Configuration
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dep_path = os.path.join(script_dir, '../../', 'conda_dependencies.yml')
    script_args = {
        '-l': args.log_level
    }
    estimator = PyTorch(
        entry_script=args.script_file,
        script_params=script_args,
        source_directory=script_dir,
        compute_target=args.cluster_name,
        distributed_training=MpiConfiguration(),
        framework_version=pytorch_version,
        use_gpu=is_gpucluster,
        conda_dependencies_file=dep_path,
        inputs=[
            dataset_train.as_named_input('subset_train'),
            dataset_test.as_named_input('subset_test')
        ]
    )

    # Set parameters for search
    param_sampling = BayesianParameterSampling({
        'learning_rate': uniform(0.05, 0.1),
        'num_epochs': choice(5, 10, 15),
        'batch_size': choice(150, 200),
        'hidden_size': choice(50, 100)
    })

    # Define multi-run configuration
    run_config = HyperDriveConfig(
        estimator=estimator,
        hyperparameter_sampling=param_sampling,
        policy=None,
        primary_metric_name='Accuracy',
        primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=args.hyperparam_runs_max,
        max_concurrent_runs=args.hyperparam_concur_max
    )

    # Define the ML experiment
    experiment = Experiment(workspace, args.experiment_name)

    # Submit the experiment
    run = experiment.submit(run_config)
    run.wait_for_completion(show_output=True)

    # Select the best run from all submitted

    # Log the best run's performance to the parent run

    # Best set of parameters found


if __name__ == '__main__':

    main(sys.argv)
