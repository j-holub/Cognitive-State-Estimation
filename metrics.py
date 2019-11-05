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


metric_compare = subparser.add_parser('metric_compare')
metric_compare.add_argument('HistoryFiles',
                       nargs='+')
metric_compare.add_argument('--metrics', '-m',
                       nargs='+',
                       choices=[
                        'acc',
                        'val_acc',
                        'loss',
                        'val_loss'
                       ],
                       default=['acc', 'val_acc'],
                       help='The metrics that should be displayed in the grouped barchart')
metric_compare.add_argument('--axisname', '-ax',
                       default='Accuracy',
                       help='The name for the Y Axis')
metric_compare.add_argument('--output','-o',
                       default='.',
                       help='Where to output the plot image')



approach_compare = subparser.add_parser('approach_compare')
approach_compare.add_argument('--face', '-f',
                               nargs='+',
                               help='Result files for the face approach')
approach_compare.add_argument('--eye', '-e',
                               nargs='+',
                               help='Result files for the eye approach')
approach_compare.add_argument('--metric', '-m',
                               default='acc',
                               choices=['acc', 'val_acc', 'loss', 'val_loss'],
                               help='The metric that should be compared')
approach_compare.add_argument('--axisname', '-ax',
                               default='Accuracy',
                               help='The name for the Y Axis')
approach_compare.add_argument('--output','-o',
                               default='.',
                               help='Where to output the plot image')



acc_distribution = subparser.add_parser('acc_dist')
acc_distribution.add_argument('HistoryFiles',
                               nargs='+',
                               help='The history files output by the training')
acc_distribution.add_argument('--metric', '-m',
                               default='acc',
                               choices=['acc', 'val_acc'],
                               help='The metric that should be compared')
acc_distribution.add_argument('--output','-o',
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
    import re
    import lib.plot as plot

    history_file = os.path.abspath(args.History)
    out_dir      = os.path.abspath(args.output)

    assert os.path.exists(out_dir)
    assert os.path.exists(history_file) \
        and os.path.isfile(history_file) \
        and os.path.splitext(history_file)[1] == '.json'


    # open the history json file
    with open(history_file, 'r') as his:
        data = json.load(his)

        # plot the relevant metricss
        if(args.Metric == 'loss'):
            plot.loss_axisplot(
                [data['loss'], data['val_loss']],
                ['Training', 'Validation'],
            )
        else:
            plot.accuracy_axisplot(
                [data['acc'], data['val_acc']],
                ['Training', 'Validation']
            )

    # get the participant's indentifier from the history file
    p = match = re.search('p\d\d', history_file).group()
    # save the graph as an
    plt.savefig(os.path.join(out_dir, '{}_{}.png'.format(p, args.Metric)))

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



elif(args.Metric == 'approach_compare'):

    assert os.path.exists(args.output) and os.path.isdir(args.output)

    import lib.plot as plot

    result_handlers = {}

    # result files for the face approach
    if(args.face):
        result_handlers['face'] = plot.ResultsHandler()
        for face_result_file in args.face:
            result_handlers['face'].add_result_file(face_result_file)

    # result files for the eye approach
    if(args.eye):
        result_handlers['eye'] = plot.ResultsHandler()
        for eye_result_file in args.eye:
            result_handlers['eye'].add_result_file(eye_result_file)

    data = [list(resh.get_metric(args.metric).values()) for resh in result_handlers.values()]
    data_labels = list(result_handlers.keys())
    labels = list(result_handlers.values())[0].get_persons()


    plot.barchart(
        labels,
        data,
        data_labels,
        args.axisname
    )

    out_file_name = '{}_{}_barchart.png'.format('_'.join(data_labels), args.axisname)

    # save the graph as an
    plt.savefig(os.path.join(args.output, out_file_name))



elif (args.Metric == 'acc_dist'):

    assert os.path.exists(args.output) and os.path.isdir(args.output)

    import lib.plot as plot

    resh = plot.ResultsHandler()

    for his in args.HistoryFiles:
        resh.add_result_file(his)

    result = np.array(list(resh.get_metric(args.metric).values()))

    plot.acc_distribution_plot([result], ['Face'])


    out_file_name = '{}_distribution.png'.format(args.metric)
    # save the graph as an
    plt.savefig(os.path.join(args.output, out_file_name))
