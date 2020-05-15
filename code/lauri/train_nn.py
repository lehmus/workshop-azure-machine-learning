import argparse
from azureml.core import Run
from collections import Counter
import logging
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.externals import joblib
import sys
import torch


class OurNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(OurNet, self).__init__()
        self.layer_1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = torch.nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out


class NewsgroupData:
    def __init__(self, data=[], target=[]):
        self.data = data
        self.target = target


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


def get_word_2_index(vocab):
    '''
    Get the index of the vocabulary
    '''

    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


def get_batch(df_data, df_target, i, batch_size, total_words, word2index):
    '''
    Get batch
    '''

    batches = []
    texts = df_data[i*batch_size:i*batch_size+batch_size]
    categories = df_target[i*batch_size:i*batch_size+batch_size]

    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
        batches.append(layer)

    return np.array(batches), np.array(categories)


def binary_accuracy(preds, y):
    '''
    Define metric
    '''

    correct = 0
    total = 0
    # round predictions to the closest integer
    _, predicted = torch.max(preds, 1)
    correct += (predicted == y).sum()
    correct2 = float(correct)
    total += y.size(0)
    acc = (correct2 / total)
    return acc


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
            'default_value': 'model.pickle',
            'description': 'Output model file path.'
        },
        'hidden_size': {
            'short_flag': 's',
            'type': int,
            'default_value': 100,
            'description': 'Number of nodes in the hidden layer'
        },
        'learning_rate': {
            'short_flag': 'r',
            'type': float,
            'default_value': 0.01,
            'description': 'Learning rate'
        },
        'batch_size': {
            'short_flag': 'b',
            'type': int,
            'default_value': 200,
            'description': 'Batch size'
        },
        'num_epochs': {
            'short_flag': 'e',
            'type': int,
            'default_value': 20,
            'description': 'Number of training epochs'
        },
        'log_level': {
            'short_flag': 'l',
            'type': str,
            'default_value': 'warning',
            'description': 'Logging output level.'
        }
    }
    args = get_arguments(argv, program_args)

    # Set up logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = getattr(logging, args.log_level, None)
    logging.basicConfig(level=log_level, format=log_format)

    # Get Azure ML run context
    run = Run.get_context()

    # Load the data from sklearn 20newsgroups

    # define datasets
    data_train = NewsgroupData()
    data_test = NewsgroupData()
    # read dataset from AML
    dataset_train = run.input_datasets['train'].to_pandas_dataframe()
    dataset_test = run.input_datasets['test'].to_pandas_dataframe()
    # convert to numpy df
    data_train.data = dataset_train.text.values
    data_test.data = dataset_test.text.values
    # convert label to int
    data_train.target = [int(value or 0) for value in dataset_train.target.values]
    data_test.target = [int(value or 0) for value in dataset_test.target.values]

    vocab = Counter()
    for text in data_train.data:
        for word in text.split(' '):
            vocab[word.lower()] += 1
    for text in data_test.data:
        for word in text.split(' '):
            vocab[word.lower()] += 1

    total_words = len(vocab)
    input_size = total_words  # Words in vocab
    num_classes = len(np.unique(data_train.target))
    # Categories: graphics, scispace and baseball

    # input [batch_size, n_labels]
    # output [max index for each item in batch, ... ,batch_size-1]
    loss = torch.nn.CrossEntropyLoss()
    input = torch.autograd.Variable(torch.randn(2, 5), requires_grad=True)
    target = torch.autograd.Variable(torch.LongTensor(2).random_(5))
    output = loss(input, target)
    output.backward()

    net = OurNet(input_size, args.hidden_size, num_classes)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    word2index = get_word_2_index(vocab)

    # Train the Model
    epoch_losses = []
    epoch_accuracy = []
    for epoch in range(args.num_epochs):
        total_batch = int(len(data_train.data) / args.batch_size)
        epoch_loss = 0
        epoch_acc = 0
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(
                data_train.data, data_train.target, i,
                args.batch_size, total_words, word2index
            )
            articles = torch.autograd.Variable(torch.FloatTensor(batch_x))
            labels = torch.autograd.Variable(torch.LongTensor(batch_y))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(articles)
            loss = criterion(outputs, labels)
            acc = binary_accuracy(outputs, labels)

            loss.backward()
            optimizer.step()

            # loss and accuracy
            epoch_loss = loss.item()
            epoch_acc = acc

        epoch_losses.append(epoch_loss / total_batch)
        epoch_accuracy.append(epoch_acc)

    # Create plot for loss percentage
    plt.plot(np.array(epoch_losses), 'r', label='Loss')
    plt.xticks(np.arange(1, (args.num_epochs+1), step=1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss Percentage')
    plt.legend(loc='upper left')
    # plt.show()
    run.log_image('Loss', plot=plt)

    # Create plot for accuracy percentage
    plt.plot(np.array(epoch_accuracy), 'b', label='Accuracy')
    plt.xticks(np.arange(1, (args.num_epochs+1), step=1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Percentage')
    plt.legend(loc='upper left')
    # plt.show()
    run.log_image('Accuracy', plot=plt)

    # Test the Model
    correct = 0
    total = 0
    total_test_data = len(data_test.target)
    batch_x_test, batch_y_test = get_batch(
        data_test.data, data_test.target, 0, total_test_data,
        total_words, word2index
    )
    articles = torch.autograd.Variable(torch.FloatTensor(batch_x_test))
    labels = torch.LongTensor(batch_y_test)
    outputs = net(articles)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    correct2 = float(correct)

    accuracy = (correct2 / total)

    # Log the metrics and parameters in AML Service
    run.log('Accuracy', float(accuracy))
    run.log('Learning rate', args.learning_rate)
    run.log('Number of epochs', args.num_epochs)
    run.log('Batch size', args.batch_size)
    run.log('Hidden layer size', args.hidden_size)

    # Save pickle file
    model_file = args.output_model_file
    model_path = os.path.join(args.output_dir, model_file)
    joblib.dump(value=outputs, filename=model_path)
    run.upload_file(name=model_file, path_or_stream=model_path)

    # Close the run
    run.complete()


if __name__ == '__main__':

    main(sys.argv)
