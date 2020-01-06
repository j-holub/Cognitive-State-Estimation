"""Deep Learning Module

This module offers various neural network architectures and utility classes for
deep learning

Classes:
    DataHandler
        This class manages training data for a neural network. It shuffles the data,
        splits it into train and test dataset and normalizes the data

Functions:
    CLitW_network()
        Network architecture proposed by Fridman et al. in 2018 in their work
        "Cognitive Load Estimation in the Wild"
    twoclass_CLitW_network()
        Same network architecture as in CLitW_network() but with only two neurons
        on the output layer
    of_network()
        A network proposed origially by Peng et al. in 2017 in their work
        "Dual Temporal Scale Convolutional Neural Network for Micro-Expression
        Recognition". It was originally used to classify micro-expressions using
        optical flow images and is adapted by us to perform cognitive load
        estimation on optical flow images
    score_regression_network()
        The same network porposed by Fridman et al. in 2018 in their work
        "Cognitive Load Estimation in the Wild", but adapted for score regression.
        It tries to estimate the score achieved by a participant on an n-back trial
        between 1-10
"""

from .datahandler import DataHandler

# networks
from .cognitiveloadinthewild import CLitW_network
from .optical_flow import of_network
from .score_regression import score_regression_network
from .two_class_conv3d_timeseries_network import twoclass_CLitW_network
