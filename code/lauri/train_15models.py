import argparse
from azureml.core import Run
import logging
import os
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
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


def benchmark(clf, inputs_train, inputs_test, labels_train, labels_test):
    '''
    Benchmark classifier performance.
    '''

    # train a model
    clf.fit(inputs_train, labels_train)

    # evaluate on test set
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

    # Extracting features from the train and test data using the same vectorizer
    inputs_train = vectorizer.fit_transform(data_train.data)
    inputs_test = vectorizer.transform(data_test.data)

    # Store models in a dictionary
    models = dict()

    # Standard classifiers
    models['Ridge Classifier'] = RidgeClassifier(tol=1e-2, solver='sag')
    models['Perceptron'] = Perceptron(max_iter=50)
    models['Passive-Aggressive'] = PassiveAggressiveClassifier(max_iter=50)
    models['kNN'] = KNeighborsClassifier(n_neighbors=10)
    models['Random forest'] = RandomForestClassifier()

    # Liblinear model & SGD model
    # Try with different regularization techniques
    for penalty in ['l1', 'l2']:
        models['LinearSVC-' + penalty] = LinearSVC(
            penalty=penalty, dual=False, tol=1e-3
        )
        models['SGDClassifier-' + penalty] = SGDClassifier(
            alpha=.0001, max_iter=50, penalty=penalty
        )

    # SGD with Elastic Net penalty
    models['Elastic-Net penalty'] = SGDClassifier(alpha=.0001, max_iter=50, penalty='elasticnet')

    # NearestCentroid without threshold
    models['NearestCentroid (aka Rocchio classifier)'] = NearestCentroid()

    # Sparse Naive Bayes classifiers
    models['Naive Bayes MultinomialNB'] = MultinomialNB(alpha=.01)
    models['Naive Bayes BernoulliNB'] = BernoulliNB(alpha=.01)
    models['Naive Bayes ComplementNB'] = ComplementNB(alpha=.1)

    # Liblinear: The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    models['LinearSVC with L1-based feature selection'] = Pipeline([
        (
            'feature_selection',
            SelectFromModel(LinearSVC(penalty='l1', dual=False, tol=1e-3))
        ),
        (
            'classification',
            LinearSVC(penalty='l2')
        )
    ])

    # Run benchmark and collect results with multiple classifiers

    # TODO: miksei päädy mihinkään lokiin? tallenna omaan logs/app.log tiedostoon?
    logging.info('Starting to train {} algorithms as child runs.'.format(len(models.keys())))
    max_score = 0.0
    for model_name in models.keys():
        # Create a child run for Azure ML logging
        child_run = run.child_run(name=model_name)
        # Train and evaluate model
        _, score = benchmark(
            models[model_name], inputs_train, inputs_test, labels_train, labels_test
        )
        if score > max_score: max_score = score
        # Write accuracy to log
        child_run.log('Accuracy', float(score))
        # Create the output directory, if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # Save pickle file
        model_file = model_name.replace(' ', '_') + '.pickle'
        model_path = os.path.join(args.output_dir, model_file)
        joblib.dump(value=models[model_name], filename=model_path)
        child_run.upload_file(name=model_file, path_or_stream=model_path)
        child_run.complete()
    run.log('Accuracy', max_score)

    # Close the run
    run.complete()


if __name__ == '__main__':

    main(sys.argv)
