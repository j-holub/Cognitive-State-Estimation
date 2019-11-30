import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def score_heatmap(scores: np.ndarray):

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

    # axislabels
    plt.xlabel('Score', fontsize=12)
