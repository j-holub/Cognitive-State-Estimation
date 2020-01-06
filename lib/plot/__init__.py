"""This submodule offers various plotting functions

Using Matplotlib this submodule provides various methods to plot data. It's important
to understand that Matplotlib uses a global state object, the plot object, which
is altered by these functions, which means they all have side effects

Classes:
    ResultsHandler
        stores multiple result files retrieved from training models on a dataset
        using Keras. Gives access to different metrics over all the result files,
        such as maximum accuracy or minimum loss of every participant

Functions:
    loss_axisplot(data, labels):
        plots the loss developement for multiple datasets
    accuracy_axisplot(data, labels):
        plots the accuracy developement for multiple datasets
    acc_distribution_plot(accuracies, labels):
        plots the distribution of for different accuracy score thresholds, the
        number of  participants the models were able to achieved at least said
        accuracy score
        Can plot multiple datsets into one plot
    barchart(labels, data, data_label, y_label):
        plots grouped barcharts for different datasets that contain values for
        a multi-user basis. One group per user with one bar per dataset
    score_heatmap(scores):
        plots a heatmap defined by scores. Can be used to plot the number of times
        a certain score was achieved for which difficulty of the n-back task
"""

from .axisplot import loss_axisplot
from .axisplot import accuracy_axisplot
from .axisplot import acc_distribution_plot
from .barchart import barchart
from .heatmap import score_heatmap

from .resultshandler import ResultsHandler
