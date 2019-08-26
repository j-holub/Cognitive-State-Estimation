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
        path to the directory storing all the files gathered from the N-Back
        experiment
    output (str, optional):
        directory where the output should be stored in.
        default: '.'
    window (int):
        window size, which means how many frames should be chunked together for
        one label
        default: 60
    crop (int):
        crop dimension for the face images extracted from the frames
"""

import argparse
import os

import lib.dataprocessing as dp


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('ExperimentData',
                     help='The directory that holds all the data recorded at the\
                           N-Back experiment')
parser.add_argument('--output', '-o', default='.',
                     help='Directory to store the output in')
parser.add_argument('--window', '-w', default=60,
                     help='The window size for a single input in frames')
parser.add_argument('--crop', '-c', default=64,
                     help='The crop size for the frames')
arguments = parser.parse_args()

# Argument Processing
exp_data_path = os.path.abspath(arguments.ExperimentData)
output_path   = os.path.abspath(arguments.output)
windowsize    = int(arguments.window)
cropsize      = int(arguments.crop)




# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isdir(exp_data_path)
assert os.path.exists(output_path) and os.path.isdir(output_path)

# retrieve the files from the experiment directory
exp_json, video_path, participant = dp.util.getExperimentInfo(exp_data_path)

assert os.path.exists(video_path) and os.path.isfile(video_path) \
    and os.path.splitext(video_path)[1] == '.mp4'
assert os.path.exists(exp_json) and os.path.isfile(exp_json) \
    and os.path.splitext(exp_json)[1] == '.json'


print('')
print('Experiment Data: {}'.format(exp_json))
print('Video: {}'.format(video_path))
print('Output Path: {}'.format(output_path))
print('Windowsize: {}'.format(windowsize))
print('Cropsize: {}'.format(cropsize))
print('Participant: {}'.format(participant))
print('')

# create the data processing objects
exp_data      = dp.ExperimentData(exp_json)
video_handler = dp.VideoHandler(video_path)
data_handler  = dp.DataHandler((windowsize, cropsize, cropsize))

print("Processing N-levels...")
# iterate over the difficulty levels 1-5
for n in range(1,6):
    print("N={}".format(n))
    # get the trial data for each difficulty level
    trials = exp_data.get_trials(n)
    # iterate over every trial
    for i, trial in enumerate(trials):
        print("  Trial {}".format(i+1))
        # get the face frames for the trial
        frames = video_handler.get_frames(trial['start'], trial['end'], cropsize)
        # store the away in the data handler with the correct label
        data_handler.add_frames(frames, n)

print("Writing data to disk...")
# write the data to disk
data_path, labels_path = data_handler.write(output_path, participant)
print("  {}".format(data_path))
print("  {}".format(labels_path))
