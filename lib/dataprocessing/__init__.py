"""Data processing module

This package offers various classes and functions to process the data gathered
from the experiments.

Classes:
    DataHandler
        Handles the processed data and stores it in the right format. It groups
        the frames into chunks of windowsize size and manages labels for those
        chunks
    ExperimentData
        Handles the JSON file created by the n-back experiment implemented in
        JsPsych. Processes the data and offers various method to access the
        relevant information
    Statistics
        Computes various statistics about the experiment data, such as average
        score and so on
    VideoHandler
        Handles the video recorded during the experiment and is able to extract
        the frames needed for the data
"""

from .datahandler    import DataHandler
from .experimentdata import ExperimentData
from .statistics     import Statistics
from .video          import VideoHandler

from .util import *
