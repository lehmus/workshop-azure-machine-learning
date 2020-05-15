'''
Train the model with the full dataset, using the hyperparameters from the best run.
'''

import argparse
from azureml.core import Workspace, Experiment
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.dataset import Dataset
from azureml.core.runconfig import MpiConfiguration
from azureml.train.dnn import PyTorch
from azureml.train.hyperdrive import HyperDriveRun
import logging
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
            'description': (
                'Script file path on the host system'
                ' (relative to the parent script directory).'
            )
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
        'log_level': {
            'short_flag': 'l',
            'type': str,
            'default_value': 'warning',
            'description': 'Logging output level. Possible values: error, warning, info, debug.'
        }
    }
    args = get_arguments(argv, program_args)

    # PyTorch library version
    pytorch_version = '1.4'

    # Set up logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = getattr(logging, args.log_level, None)
    logging.basicConfig(level=log_level, format=log_format)

    # Load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())

    # Check if GPU computing is available
    is_gpucluster = ('gpu' in args.cluster_name or args.cluster_name == 'local')

    # Retrieve datasets used for training
    dataset_train = Dataset.get_by_name(workspace, name='newsgroups_subset_train')
    dataset_test = Dataset.get_by_name(workspace, name='newsgroups_subset_test')


    # Create compute target if not present

    # Choose a name for your CPU cluster
    # cluster_name = "fullmodelcomputegpu"
    # Verify that cluster does not exist already
    try:
        cluster = ComputeTarget(workspace=workspace, name=args.cluster_name)
        logging.info('Using existing compute cluster \'' + args.cluster_name + '\'')
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='Standard_NC12', max_nodes=8
        )
        logging.info('Creating new compute cluster \'' + args.cluster_name + '\'')
        cluster = ComputeTarget.create(workspace, args.cluster_name, compute_config)
        cluster.wait_for_completion(show_output=True)


    # Get best parameters hyperdrive run

    # Define the ML experiment
    experiment = Experiment(workspace, args.experiment_name)
    # Get all the runs in the experiment
    generator = experiment.get_runs(type=None, tags=None, properties=None, include_children=False)
    run = next(generator)
    # Select the last run
    parent = HyperDriveRun(experiment, run_id=run.id)
    # Select the best run from all submitted
    best_run = parent.get_best_run_by_primary_metric()
    # best_run_metrics = best_run.get_metrics()
    # Best set of parameters found
    parameter_values = best_run.get_details()['runDefinition']['arguments']
    best_parameters = dict(zip(parameter_values[::2], parameter_values[1::2]))
    best_model_parameters = best_parameters.copy()

    # Define a final training run with model's best parameters
    script_args = {
        '-l': args.log_level
    }
    for param in best_model_parameters.keys():
        script_args[param] = best_model_parameters[param]
    model_est = PyTorch(
        entry_script=args.script_file,
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        script_params=script_args,
        compute_target=workspace.compute_targets[args.cluster_name],
        distributed_training=MpiConfiguration(),
        framework_version=pytorch_version,
        use_gpu=is_gpucluster,
        pip_packages=[
            'numpy==1.15.4',
            'pandas==0.23.4',
            'scikit-learn==0.20.1',
            'scipy==1.0.0',
            'matplotlib==3.0.2',
            'utils==0.9.0',
        ],
        inputs=[
            dataset_train.as_named_input('train'),
            dataset_test.as_named_input('test')
        ]
    )

    # Define the ML experiment
    experiment = Experiment(workspace, args.experiment_name)
    # Submit the experiment
    run = experiment.submit(model_est)
    _ = run.wait_for_completion(show_output=True, wait_post_processing=True)


if __name__ == '__main__':

    main(sys.argv)
