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
    'twoclass_clitw': deepl.twoclass_CLitW_network,
    'score': deepl.score_regression_network
}
# set the network according to the input argument provided
net = networks[network]()

# compile the network
net.compile(optimizer=optimizers.sgd(lr=0.01, momentum=0.9),
              loss='mean_squared_error',
              metrics=['accuracy'])


best_acc_path = os.path.join(out_dir, '{}_bestacc_{}_{}_model.h5'.format(
                    network,
                    datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                    suffix)
                )
best_acc_checkpoint = ModelCheckpoint(best_acc_path,
                    monitor='acc',
                    verbose=1,
                    save_best_only=True,
                    mode='max'
                    )
best_valacc_path = os.path.join(out_dir, '{}_bestvalacc_{}_{}_model.h5'.format(
                      network,
                      datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                      suffix)
                   )
best_valacc_checkpoint = ModelCheckpoint(best_valacc_path,
                       monitor='val_acc',
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

# save the history if specified
with open(os.path.join(his_dir, '{}_e{}_{}history.json'.format(network, epochs, suffix)), 'w') as his_file:
    json.dump(history.history, his_file)
