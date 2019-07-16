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
"""

from .experimentdata import ExperimentData
from .video import VideoHandler
