import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='Metric')

# confusion matrix parser
confm = subparser.add_parser('confmat')
confm.add_argument('Data',
                    help='The data to input to the network')
confm.add_argument('Labels',
                    help='The labels to check the predictions against')
confm.add_argument('Model',
                    help='The trained model as a .h5 file')
confm.add_argument('--validation-split', '-vs',
                    default=0.9,
                    help='The validation split that should be used to seperate \
                          training from test data')
confm.add_argument('--output', '-o',
                    default='.',
                    help='Where to output the confusion matrix image')


loss = subparser.add_parser('loss')
loss.add_argument('History',
                   help='History output file from the network training')
loss.add_argument('--output', '-o',
                   default='.',
                   help='Where to output the plot image')


loss = subparser.add_parser('accuracy')
loss.add_argument('History',
                   help='History output file from the network training')
loss.add_argument('--output', '-o',
                   default='.',
                   help='Where to output the plot image')

args = parser.parse_args()


# confusion matrix part
if(args.Metric == 'confmat'):

    import keras.models
    import lib.deeplearning as dp
    import lib.statistics   as stat

    data_file  = os.path.abspath(args.Data)
    label_file = os.path.abspath(args.Labels)
    model_file = os.path.abspath(args.Model)
    val_split  = float(args.validation_split)
    out_dir    = os.path.abspath(args.output)

    assert val_split > 0.1 and val_split < 1
    assert os.path.exists(out_dir)


    # load the data
    dh = dp.DataHandler(data_file, label_file, val_split)
    # load the model
    model = keras.models.load_model(model_file)
    # retrieve the test data
    test_x, test_y = dh.test_data()
    # compute the confusion matrix
    confmat = stat.confusion_mat(model, test_x[:50], test_y[:50])

    # plot it
    fig, ax = plt.subplots()
    ax.matshow(confmat)

    # set the values to the cells
    for (i,j), v in np.ndenumerate(confmat):
        ax.text(j,i, '{:0.2f}'.format(v), ha='center', va='center')

    plt.savefig(os.path.join(out_dir, 'confmat.png'))


# loss plot
elif (args.Metric == 'loss'):

    history_file = os.path.abspath(args.History)
    out_dir      = os.path.abspath(args.output)

    assert os.path.exists(out_dir)
    assert os.path.exists(history_file) \
        and os.path.isfile(history_file) \
        and os.path.splitext(history_file)[1] == '.json'

    with open(history_file, 'r') as his:
        data = json.load(his)

        # get the loss metric
        train_loss = data['loss']
        val_loss   = data['val_loss']

        assert len(train_loss) == len(val_loss)

        # epochs for the x axis
        x_axis = np.arange(len(train_loss), dtype=np.uint8)+1

        plt.plot(x_axis, train_loss, label='loss')
        plt.plot(x_axis, val_loss, label='validation loss')

        plt.legend()

        plt.savefig('loss.png')


# accuracy plot
elif (args.Metric == 'accuracy'):

    history_file = os.path.abspath(args.History)
    out_dir      = os.path.abspath(args.output)

    assert os.path.exists(out_dir)
    assert os.path.exists(history_file) \
        and os.path.isfile(history_file) \
        and os.path.splitext(history_file)[1] == '.json'

    with open(history_file, 'r') as his:
        data = json.load(his)

        # get the accuracy metric
        train_accuracy = data['acc']
        val_accuracy   = data['val_acc']

        assert len(train_accuracy) == len(val_accuracy)

        # epochs for the x axis
        x_axis = np.arange(len(train_accuracy), dtype=np.uint8)+1

        plt.plot(x_axis, train_accuracy, label='accuracy')
        plt.plot(x_axis, val_accuracy, label='validation accuracy')

        plt.legend()

        plt.savefig('accuracy.png')
