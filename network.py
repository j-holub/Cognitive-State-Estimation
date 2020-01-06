""" This script launches a neural network training session using Keras and Tensorflow

This script serves as a basic interface to all the neural networks provided in this
project. It is used to loat a network with the training and validation data,
performs the training over a set number of epochs and saves the best epochs for
the training and valiation set as a model and outputs a history file.

There are 4 different networks available
    clitw
        Originally proposed by Fridman et al. in 2018 in their work "Cognitive
        Load Estimation in the Wild". The network is adapted to have 5 neurons
        on the output layer for the 5 n classes, It is used for cognitive load
        estimation from time series of face or eye images
    score
        The same netwok as clitw as porposed by Fridman et al., but it features
        a single neuron on the output layer for a regression task, to estimate
        the score achieved on the n-back trial from the video data
    opticalflow
        Originally proposed by Peng at al. in 2019 in their work "Dual Temporal
        Scale Convolutional Neural Network for Micro-Expression Recognition", this
        network was intended to be used for micro-expression classification. we
        adapted the network to use it for Cognitive Load Estimation from timeseries
        of facial optical flow images
    twoclass_clitw
        Same network as clitw but with only two neurons on the output layer for
        a 2 class classification task

...

Arguments:
    Network (str):
        One of 'clitw', 'score', 'opticalflow' or 'twoclass_clitw
    TrainingFeatures(str):
        Path to the .npy file containing the features for the training set
    TrainingLabels(str):
        Path to the .npy file containing the labels for the training set
    ValidationFeatures(str):
        Path to the .npy file containing the features for the validation set
    ValidationLabels(str):
        Path to the .npy file containing the features for the validation set
    --epochs, -e (optional, int):
        Number of epochs to train
        default: 10
    --save-model, -sm (optional, str):
        path where the models are saved to
        default: '.'
    --save-history, -sh (optional, str):
        path where the history file is saved to
        default: '.'
    --suffix, -sf (optional, str)
        any arbitrary string that is added to the end of the output files
"""


import argparse
import datetime
import json
import os

import keras.utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np

import lib.deeplearning as deepl


parser = argparse.ArgumentParser()
parser.add_argument('Network',
                     choices=[
                        'clitw',
                        'score',
                        'opticalflow',
                        'twoclass_clitw'
                     ],
                     help='The neural network to use')
parser.add_argument('TrainingFeatures',
                     help='Numpy file containing the training features')
parser.add_argument('TrainingLabels',
                     help='Numpy file containing the training labels')
parser.add_argument('ValidationFeatures',
                     help='Numpy file containing the validation features')
parser.add_argument('ValidationLabels',
                     help='Numpy file containing the validation labels')
parser.add_argument('--epochs', '-e', default=10,
                     help='Number of epochs to train')
parser.add_argument('--save-model', '-sm',
                     default='.',
                     help='Save the keras model to the path specified')
parser.add_argument('--save-history', '-sh',
                     default='.',
                     help='Save the training history to the path specified')
parser.add_argument('--suffix', '-sf',
                     help='Suffix to add to the model and history file')
arguments = parser.parse_args()


network             = arguments.Network
train_features_file = os.path.abspath(arguments.TrainingFeatures)
train_labels_file   = os.path.abspath(arguments.TrainingLabels)
valid_features_file = os.path.abspath(arguments.ValidationFeatures)
valid_labels_file   = os.path.abspath(arguments.ValidationLabels)
epochs              = int(arguments.epochs)
out_dir             = None if not arguments.save_model   else os.path.abspath(arguments.save_model)
his_dir             = None if not arguments.save_history else os.path.abspath(arguments.save_history)
suffix              = '' if not arguments.suffix else '{}_'.format(arguments.suffix)


assert os.path.exists(train_features_file) \
        and os.path.isfile(train_features_file) \
        and os.path.splitext(train_features_file)[1] == '.npy'
assert os.path.exists(train_labels_file) \
        and os.path.isfile(train_labels_file) \
        and os.path.splitext(train_labels_file)[1] == '.npy'
assert os.path.exists(valid_features_file) \
        and os.path.isfile(valid_features_file) \
        and os.path.splitext(valid_features_file)[1] == '.npy'
assert os.path.exists(valid_labels_file) \
        and os.path.isfile(valid_labels_file) \
        and os.path.splitext(valid_labels_file)[1] == '.npy'
if(out_dir):
    assert os.path.exists(out_dir)
if(his_dir):
    assert os.path.exists(his_dir)
assert epochs > 0


# load the data into the datahandler
datahandler = deepl.DataHandler(
                train_features_file,
                train_labels_file,
                valid_features_file,
                valid_labels_file,
                regression=True if network == 'score' else False
)

networks = {
    'clitw': deepl.CLitW_network,
    'opticalflow': deepl.of_network,
    'twoclass_clitw': deepl.twoclass_CLitW_network,
    'score': deepl.score_regression_network
}
# set the network according to the input argument provided
net = networks[network]()

# compile the network
net.compile(optimizer=optimizers.sgd(lr=0.01, momentum=0.9),
              loss='mean_squared_error',
              metrics=['accuracy'])

# date when the session was started
session_start = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# model with the best accuracy on the training set
best_acc_path = os.path.join(out_dir, '{}_bestacc_{}_{}_model.h5'.format(
                    network,
                    session_start,
                    suffix)
                )
best_acc_checkpoint = ModelCheckpoint(best_acc_path,
                    monitor='accuracy',
                    verbose=1,
                    save_best_only=True,
                    mode='max'
                    )

# model with the best accuracy on the validation set
best_valacc_path = os.path.join(out_dir, '{}_bestvalacc_{}_{}_model.h5'.format(
                      network,
                      session_start,
                      suffix)
                   )
best_valacc_checkpoint = ModelCheckpoint(best_valacc_path,
                       monitor='val_accuracy',
                       verbose=1,
                       save_best_only=True,
                       mode='max'
                       )

callbacks_list = [best_acc_checkpoint, best_valacc_checkpoint]


# get the training data from the data handler
train_x, train_y = datahandler.train_data()
# train the model
history = net.fit(train_x,
        train_y,
        epochs=epochs,
        batch_size=5,
        validation_data=datahandler.test_data(),
        callbacks=callbacks_list
)

# transform the history object to floats to make it JSON serialisable
output = history.history
for key in output.keys():
    output[key] = [float(x) for x in output[key]]

# save the history if specified
with open(os.path.join(his_dir, '{}_e{}_{}_{}history.json'.format(network, epochs, session_start, suffix)), 'w') as his_file:
    json.dump(output, his_file)
