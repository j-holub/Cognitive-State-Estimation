"""Data processing module

This package offers various classes and functions to process the data gathered
from the experiments.

Classes:
    ExperimentData
        Handles the JSON file created by the n-back experiment implemented in
        JsPsych. Processes the data and offers various method to access the
        relevant information
    VideoHandler
        Handles the video recorded during the experiment and is able to extract
        the frames needed for the data
    DataHandler
        Handles the processed data and stores it in the right format. It groups
        the frames into chunks of windowsize size and manages labels for those
        chunks
"""

from .experimentdata import ExperimentData
from .video          import VideoHandler
from .datahandler    import DataHandler
