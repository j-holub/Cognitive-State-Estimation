import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import lib.deeplearning as dp
import lib.statistics   as stat

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

args = parser.parse_args()


# confusion matrix part
if(args.Metric == 'confmat'):

    import keras.models

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
