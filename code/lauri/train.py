'''
Train and evaluate a classification model.
'''

import argparse
from azureml.core import Run
import logging
from optparse import OptionParser
import os
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
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


def benchmark(clf, inputs_train, inputs_test, labels_train, labels_test, name=''):
    '''
    Benchmark classifier performance.
    '''

    # Train model
    clf.fit(inputs_train, labels_train)

    # Evaluate model on test set
    pred = clf.predict(inputs_test)
    score = metrics.accuracy_score(labels_test, pred)
    clf_descr = str(clf).split('(')[0]

    return clf_descr, score


def main(argv):

    # Define and read program input arguments
    program_args = {
        'output_dir': {
            'short_flag': 'o',
            'type': str,
            'default_value': 'outputs',
            'description': 'Output folder path on the host system.'
        },
        'output_model_file': {
            'short_flag': 'f',
            'type': str,
            'default_value': 'model.pkl',
            'description': 'Output model file path.'
        },
        'log_level': {
            'short_flag': 'l',
            'type': str,
            'default_value': 'warning',
            'description': 'Logging output level.'
        }
    }
    args = get_arguments(argv, program_args)

    # Define categories to extract from the 20newsgroup data from sklearn
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

    # Set up logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = getattr(logging, args.log_level, None)
    logging.basicConfig(level=log_level, format=log_format)

    # Get Azure ML run context
    run = Run.get_context()

    # Load the data from sklearn 20newsgroups
    data_train = fetch_20newsgroups(
        subset='train', categories=categories, shuffle=True, random_state=42
    )
    data_test = fetch_20newsgroups(
        subset='test', categories=categories, shuffle=True, random_state=42
    )

    # Split a training set and a test set
    labels_train, labels_test = data_train.target, data_test.target

    # Extracting features from the training data using a sparse vectorizer
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, stop_words='english'
    )

    # Extracting features from the train and test data using the vectorizer
    inputs_train = vectorizer.fit_transform(data_train.data)
    inputs_test = vectorizer.transform(data_test.data)

    # Select model to benchmark (In the lab we choose for Random Forest,
    # but any classification model from sklearn can be chosen)
    clf = RandomForestClassifier()
    name = 'Random forest'

    # Run benchmark and collect results from the selected mdodel
    logging.info('Starting to train algorithm \' ' + name + '\'.')
    _, acc = benchmark(clf, inputs_train, inputs_test, labels_train, labels_test, name)
    run.log('Accuracy', float(acc))

    # Create the output directory, if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save pickle file
    model_file = args.output_model_file
    model_path = os.path.join(args.output_dir, model_file)
    joblib.dump(value=clf, filename=model_path)
    run.upload_file(name=model_file, path_or_stream=model_path)

    # Close the run
    run.complete()


if __name__ == '__main__':

    main(sys.argv)
