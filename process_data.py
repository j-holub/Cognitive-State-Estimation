"""Processes the data of one participant and outputs it as .npy files

The script reads the video and the JSON file created by the JsPsych N-Back Experiment
and extracts the frames belonging to the n-back trials with the correct label of
difficulty. It extracts the face in the frames, transforms the image to grayscale
and resizes it to 128x128

The output are two .npy files, one containing the input data, which consists of
chunks of frames according to the windowsize.
Its dimension is (chunks, windowsize, 128, 128)

Parameters:
    ExperimentData (str):
        path to the JSON file created by the JsPsych N-Back Experiment
    Video (str):
        path to the video file recorded in the experiment
    output (str, optional):
        directory where the output should be stored in.
        default: '.'
    window (int):
        window size, which means how many frames should be chunked together for
        one label
        default: 60
    participant (str):
        name of the participant which is added to the output filename.
        default: 'p'
"""

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
parser.add_argument('--output', '-o', default='.',
                     help='Directory to store the output in')
parser.add_argument('--window', '-w', default=60,
                     help='The window size for a single input in frames')
parser.add_argument('--participant', '-p', default='p',
                     help='The participant name of that dataset')
arguments = parser.parse_args()

# Argument Processing
exp_data_path = os.path.abspath(arguments.ExperimentData)
video_path    = os.path.abspath(arguments.Video)
output_path   = os.path.abspath(arguments.output)
windowsize    = int(arguments.window)
participant   = arguments.participant

# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isfile(exp_data_path)
assert os.path.exists(video_path) and os.path.isfile(video_path) \
    and os.path.splitext(video_path)[1] == '.mp4'
assert os.path.exists(output_path) and os.path.isdir(output_path)

# create the data processing objects
exp_data      = dp.ExperimentData(exp_data_path)
video_handler = dp.VideoHandler(video_path)
data_handler  = dp.DataHandler(windowsize)

# iterate over the difficulty levels 1-5
for n in range(1,6):
    # get the trial data for each difficulty level
    trials = exp_data.get_trials(n)
    # iterate over every trial
    for trial in trials:
        # get the face frames for the trial
        frames = video_handler.get_frames(trial['start'], trial['end'])
        # store the away in the data handler with the correct label
        data_handler.add_frames(frames, n)

# write the data to disk
data_handler.write(output_path, participant)
