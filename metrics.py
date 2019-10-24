"""This script plots different metrics from the data gathered by training the networks

The script takes data as input, for example the loss or accuracy timeseries and
plots it as a graph. It can also plot the confusion matrix if the model, data
and labels are provided

The following metrics are available
    confmat
        Plots the confusion matrix for the given model and data/labels
    loss
        Plots the loss from the output provided by the Keras history object
    accuracy
        Plots the accuracy from the output provided by the Keras history object

...

Parameters
    Metric (str):
        Which metric shold to compute
            confmat: confusion matrix
            loss: loss time series for epochs
            accuracy: accuracy time series for epochs

    loss/accuracy:
        History (str):
            path to the history json file output by training the neural networks
        output (str, optional):
            path where to output the plot graphic
            default: .

    confmat:
        Data (str):
            data that is input to the network to predict the label for, a .npy
            file created by the process_data.py script
        Labels (str):
            ground truth labels to check the predictions against to compute the
            confusion matrix
        Model (str):
            model file created by keras. Has to be in .h5 format
        validation-split (float, optional):
            How to split the data into train and test data. Only the test data
            is used to compute the confusion matrix. Should be the same as used
            when training the network
            default: 0.9
        output (str, optional):
            path where to output the plot graphic
            default: .
"""


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

# loss
loss = subparser.add_parser('loss')
loss.add_argument('History',
                   help='History output file from the network training')
loss.add_argument('--output', '-o',
                   default='.',
                   help='Where to output the plot image')


accuracy = subparser.add_parser('accuracy')
accuracy.add_argument('History',
                       help='History output file from the network training')
accuracy.add_argument('--output', '-o',
                       default='.',
                       help='Where to output the plot image')


barchart = subparser.add_parser('metric_compare')
barchart.add_argument('HistoryFiles',
                       nargs='+')
barchart.add_argument('--metrics', '-m',
                       nargs='+',
                       choices=[
                        'acc',
                        'val_acc',
                        'loss',
                        'val_loss'
                       ],
                       default=['acc', 'val_acc'],
                       help='The metrics that should be displayed in the grouped barchart')
barchart.add_argument('--axisname', '-ax',
                       default=['Accuracy'],
                       help='The name for the Y Axis')
barchart.add_argument('--output','-o',
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
elif (args.Metric == 'loss' or args.Metric == 'accuracy'):
    import lib.plot as plot

    history_file = os.path.abspath(args.History)
    out_dir      = os.path.abspath(args.output)

    assert os.path.exists(out_dir)
    assert os.path.exists(history_file) \
        and os.path.isfile(history_file) \
        and os.path.splitext(history_file)[1] == '.json'

    # choose the metrics according to the arguments given
    metrics = ['loss', 'val_loss'] \
              if args.Metric == 'loss' \
              else ['acc', 'val_acc']

    # open the history json file
    with open(history_file, 'r') as his:
        data = json.load(his)

        # plot the relevant metricss
        plot.axisplot(
            [data[met] for met in metrics],
            metrics
        )

    # save the graph as an
    plt.savefig(os.path.join(out_dir, '{}.png'.format(args.Metric)))

# barchart
elif(args.Metric == 'metric_compare'):

    assert os.path.exists(args.output) and os.path.isdir(args.output)

    import lib.plot as plot

    results_files = [os.path.abspath(file) for file in args.HistoryFiles]

    resh = plot.ResultsHandler()

    for res_file in results_files:
        resh.add_result_file(res_file)

    metrics = {}
    for met in args.metrics:
        metrics[met] = list(resh.get_metric(met).values())

    persons = resh.get_persons()

    plot.barchart(
        persons,
        [values for values in metrics.values()],
        [met for met in metrics.keys()],
        args.axisname
    )

    out_file_name = '{}_barchart.png'.format('_'.join(args.metrics))

    # save the graph as an
    plt.savefig(os.path.join(args.output, out_file_name))
