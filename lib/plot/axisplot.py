import matplotlib.pyplot as plt
import numpy as np

from .util import colours, clean_plot


def loss_axisplot(data: list, labels: list):
    assert len(data) == len(labels)

    fig, ax = plt.subplots()

    clean_plot(ax)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # epochs on the x axis
    x_axis = np.arange(len(data[0]))+1

    # add every data entry with the respective label
    for i in range(len(data)):
        plt.plot(x_axis, data[i], label=labels[i], color=colours()[i+2])

    loc, _ = plt.yticks()

    for y in loc[1:-1]:
        plt.plot(
            np.arange(len(data[0]))+1,
            [y] * len(data[0]),
            "--",
            lw=0.5,
            color="black",
            alpha=0.3
        )

    plt.legend(loc='upper right')



def accuracy_axisplot(data: list, labels: list):
    assert len(data) == len(labels)

    fig, ax = plt.subplots()

    clean_plot(ax)

    # epochs on the x axis
    x_axis = np.arange(len(data[0]))+1

    # add the horizontal dashed lines
    for y in np.arange(0, 1, 0.1):
        plt.plot(
            np.arange(len(data[0]))+1,
            [y] * len(data[0]),
            "--",
            lw=0.5,
            color="black",
            alpha=0.3
        )

    # force the y axis between 0 and 1 for the accuracy
    plt.ylim(0,1)

    # plot accuracy labels from 0.2 to 0.8
    plt.yticks(np.arange(0.2,1,0.2), fontsize=10)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    # add every data entry with the respective label
    for i in range(len(data)):
        plt.plot(x_axis, data[i], label=labels[i], color=colours()[i])

    plt.legend(loc='upper right')


def single_axisplot(data: np.array, x_axis: np.array):

    plt.plot(data, x_axis, 'o-', color=colours()[i])
