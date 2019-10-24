import math

import matplotlib.pyplot as plt
import numpy as np

def barchart(
        labels: list,
        data: list,
        data_label: list,
        y_label: str
    ):

    assert len(data) == len(data_label)

    # get the positions for the bars
    x = np.arange(len(labels))

    # width of a bar
    width=0.3

    # create the plot
    fig, ax = plt.subplots()


    # calculate the spacing for the bars
    if len(data) % 2 == 0:
        border = (len(data)/2)-0.5
        bar_offset = width * np.linspace(-border, border, num=len(data))
    else:
        border = math.floor(len(data)/2)
        bar_offset = width * np.linspace(-border, border, num=len(data))


    # plot the bars on the correct positions
    for d, lbl, offs in zip(data, data_label, bar_offset):
        ax.bar(x+offs, d, width, label=lbl)


    # set some options on the axis
    ax.autoscale(enable=True)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # rotate the labels
    fig.autofmt_xdate()
