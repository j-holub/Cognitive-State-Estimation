""" A module that provides statistical metrics for the data retrieved by the
n-back experiment and training the models with the recorded data

Classes:
    Statistic
        loads multiple experiment files for the different participants and provides
        methods to retrieve different statistical metrics from the data

Functions:
    confusion_mat(model, evaluation_data, evaluation_labels)
        computes a confusion matrix using predictions computed the evaluation
        data, using the trained model and the ground truth labels 
"""

from .confusionmatrix import confusion_mat
from .statistic import Statistic
