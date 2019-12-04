import math

import matplotlib.pyplot as plt
import numpy as np

from .util import colours, clean_plot

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

    # clean plot styling
    clean_plot(ax)


    for y in np.arange(0, 1, 0.1):
        plt.plot(x, [y] * len(x), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(bottom=True, left=False, top=False, right=False)
    plt.xticks(x, fontsize=8)
    plt.yticks(np.arange(0.2,1,0.2), fontsize=10)
    plt.ylim(0,1)



    # calculate the spacing for the bars
    if len(data) % 2 == 0:
        border = (len(data)/2)-0.5
        bar_offset = width * np.linspace(-border, border, num=len(data))
    else:
        border = math.floor(len(data)/2)
        bar_offset = width * np.linspace(-border, border, num=len(data))


    # plot the bars on the correct positions
    for i, (d, lbl, offs) in enumerate(zip(data, data_label, bar_offset)):
        ax.bar(x+offs, d, width, label=lbl, color=colours()[i])


    # set some options on the axis
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')

    plt.ylabel(y_label)

    # rotate the labels
    fig.autofmt_xdate()
