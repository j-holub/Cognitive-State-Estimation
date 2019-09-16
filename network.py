import argparse
import json
import os

import keras.utils
from keras import optimizers
import numpy as np

import lib.deeplearning as deepl


parser = argparse.ArgumentParser()
parser.add_argument('Network',
                     choices=['clitw'],
                     help='The neural network to use')
parser.add_argument('Features',
                     help='Numpy file containing the features')
parser.add_argument('Labels',
                     help='Numpy file containing the labels')
parser.add_argument('--epochs', '-e', default=10,
                     help='Number of epochs to train')
parser.add_argument('--validation-split', '-vs', default=0.9,
                     help='Split ratio between train and test data')
parser.add_argument('--save-model', '-sm',
                     help='Save the keras model to the path specified')
parser.add_argument('--save-history', '-sh',
                     help='Save the training history to the path specified')
arguments = parser.parse_args()


network       = arguments.Network
features_file = os.path.abspath(arguments.Features)
labels_file   = os.path.abspath(arguments.Labels)
epochs        = int(arguments.epochs)
val_split     = float(arguments.validation_split)
out_dir       = os.path.abspath(arguments.save_model)
his_dir       = os.path.abspath(arguments.save_history)

assert os.path.exists(features_file) \
        and os.path.isfile(features_file) \
        and os.path.splitext(features_file)[1] == '.npy'
assert os.path.exists(labels_file) \
        and os.path.isfile(labels_file) \
        and os.path.splitext(labels_file)[1] == '.npy'
assert os.path.exists(out_dir)
assert os.path.exists(his_dir)
assert epochs > 0
assert val_split>0 and val_split <1

# load the data into the datahandler
datahandler = deepl.DataHandler(features_file, labels_file, val_split)

networks = {
    'clitw': deepl.CLitW_network
}
# set the network according to the input argument provided
net = networks[network]()

# compile the network
net.compile(optimizer=optimizers.sgd(lr=0.01, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# get the training data from the data handler
train_x, train_y = datahandler.train_data()
# train the model
history = net.fit(train_x,
        train_y,
        epochs=epochs,
        batch_size=5,
        validation_data=datahandler.test_data()
)

# save the model if specified
if(out_dir):
    net.save(os.path.join(out_dir, '{}_e{}_model.h5'.format(network, epochs)))

# save the history if specified
if(his_dir):
    with open(os.path.join(his_dir, '{}_e{}_history.json'.format(network, epochs)), 'w') as his_file:
        json.dump(history.history, his_file)
