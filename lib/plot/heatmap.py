import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def score_heatmap(scores: np.ndarray):
    """ Plots a heatmap showing which scores were achieved how often for the
        5 different difficulty levels of the n-back experiment

        This function uses the heatmap function from the seaborn library

        This function used matplotlib and sets the state for the pyplot object.
        This means this function has side-effects

        scores (np.ndarray):
            a two-dimensional numpy array that contains the heatmap data
    """

    # plot the heatmap
    ax = sns.heatmap(scores.astype(int),
        annot=True,
        fmt='d',
        linewidths=0.5,
        square=True,
        cbar=False,
        cmap=plt.cm.Blues
    )

    # set the ticks for the labels
    ax.set_yticklabels(range(1,6))
    ax.set_xticklabels(range(1,11))
