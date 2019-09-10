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
"""

from .datahandler import DataHandler

# networks
from .cognitiveloadinthewild import CLitW_network
