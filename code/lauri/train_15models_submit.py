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

    # load Azure ML workspace
    workspace = Workspace.from_config(auth=AzureCliAuthentication())

    # Define Run Configuration
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dep_path = os.path.join(script_dir, '../../', 'conda_dependencies.yml')
    script_args = {
        '-l': args.log_level
    }
    est = Estimator(
        entry_script=args.script_file,
        script_params=script_args,
        source_directory=script_dir,
        compute_target='local',
        conda_dependencies_file=dep_path,
        use_docker=False
    )

    # Define the ML experiment
    experiment = Experiment(workspace, args.experiment_name)

    # Submit experiment run, if compute is idle, this may take some time
    run = experiment.submit(est)

    # wait for run completion of the run, while showing the logs
    run.wait_for_completion(show_output=True)

    # Select best model from run
    max_run_id = None
    max_accuracy = None

    for run in experiment.get_runs():
        run_metrics = run.get_metrics()
        run_details = run.get_details()
        if 'Accuracy' not in run_metrics.keys(): continue
        accuracy = run_metrics['Accuracy']
        run_id = run_details['runId']
        if max_accuracy is None:
            max_accuracy = accuracy
            max_run_id = run_id
        elif accuracy > max_accuracy:
            max_accuracy = accuracy
            max_run_id = run_id

    print('Best run_id: ' + max_run_id)
    print('Best run_id accuracy: ' + str(max_accuracy))


if __name__ == '__main__':

    main(sys.argv)
