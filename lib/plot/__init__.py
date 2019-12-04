"""This submodule offers various plotting functions

Using Matplotlib this submodule provides various methods to plot data. It's important
to understand that Matplotlib uses a global state object, the plot object, which
is altered by these functions, which means they all have side effects

Functions:
    axisplot(data, labels):
        plots a standard X and Y plot using the data for the Y axis and the epochs
        for the X axis. For each data series a label has to be provided.
        Plots every pair of data and label into one single plot.
"""

from .axisplot import loss_axisplot
from .axisplot import accuracy_axisplot
from .axisplot import acc_distribution_plot
from .axisplot import single_axisplot
from .barchart import barchart
from .heatmap import score_heatmap

from .resultshandler import ResultsHandler
