'''
Training submitter

Facilitates (remote) training execution through the Azure ML service.
'''

import argparse
from azureml.core import Workspace, Experiment
from azureml.train.estimator import Estimator
from azureml.core.authentication import AzureCliAuthentication
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
            'description': 'Logging output level.'
        }
    }
    args = get_arguments(argv, program_args)

    # Load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())

    # Define Run Configuration
    script_args = {
        '-l': args.log_level
    }
    est = Estimator(
        entry_script=args.script_file,
        script_params=script_args,
        source_directory=os.path.dirname(os.path.realpath(__file__)),
        compute_target='local',
        conda_packages=[
            'pip==20.0.2'
        ],
        pip_packages=[
            'numpy==1.15.4',
            'pandas==0.23.4',
            'scikit-learn==0.20.1',
            'scipy==1.0.0',
            'matplotlib==3.0.2',
            'utils==0.9.0'
        ],
        use_docker=False
    )

    # Define the ML experiment
    experiment = Experiment(workspace, args.experiment_name)

    # Submit experiment run, if compute is idle, this may take some time')
    run = experiment.submit(est)

    # Wait for run completion of the run, while showing the logs
    run.wait_for_completion(show_output=True)


if __name__ == '__main__':

    main(sys.argv)
