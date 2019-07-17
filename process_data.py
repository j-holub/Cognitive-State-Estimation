import argparse
import os

import lib.dataprocessing as dp

import cv2

# Argument Parsing

parser = argparse.ArgumentParser()
parser.add_argument('ExperimentData',
                     help='The JSON file produced by the JsPsych N-Back Experiment')
parser.add_argument('Video',
                     help='The video file recorded at the experiment')
arguments = parser.parse_args()

# path to the JSON file created by the JsPsych N-Back Experiment
exp_data_path = os.path.abspath(arguments.ExperimentData)
video_path    = os.path.abspath(arguments.Video)


# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isfile(exp_data_path)
assert os.path.exists(video_path) and os.path.isfile(video_path) \
    and os.path.splitext(video_path)[1] == '.mp4'
