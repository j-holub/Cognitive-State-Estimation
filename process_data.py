"""Processes the data of one participant and outputs it as .npy files

The script reads the video and the JSON file created by the JsPsych N-Back Experiment
and extracts the frames belonging to the n-back trials with the correct label of
difficulty.

Depending on the processing method chosen it does one of the following:
    face
        it extracts the face in every frame, transforms it to a grayscale image
        and outputs them as chunks of time series
    eye
        it extracts the right eye in every frame, transforms it to a grayscale image
        and outputs them as chunks of time series
    opticalflow
        it extracts the face in every frame and computes the optical flow image
        for every consecutive couple of two frames. It outputs them as chunks
        of time series

The output are two .npy files, one containing the input data, which consists of
chunks of frames according to the windowsize.
For face and eye time series its dimension is (chunks, windowsize, crop, crop)
For optical flow time series its dimension is (chunks, windowsize, crop, crop, 3)

---------------

Parameters:
    ProcessingMethod (str):
        The way the data should be processed
            face:        time series of grayscale face images
            eye:         time series of grayscale eye images
            opticalflow: time series of optical flow images
    ExperimentData (str):
        path to the directory storing all the files gathered from the N-Back
        experiment
    ground-truth (str, optional):
        whether the n level or the score should be used as gt
        default: 'n'
    output (str, optional):
        directory where the output should be stored in.
        default: '.'
    window (int, optional):
        window size, which means how many frames should be chunked together for
        one label
        default: 60
    crop (int, optional):
        crop dimension for the face images extracted from the frames
        default: 64
    subsample (int, optional):
        how many frames to subsample the time series data by, n means that
        every n-th frame is combined to one chunk
        default: 1
"""

import argparse
import os

import lib.dataprocessing as dp


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('ProcessingMethod',
                     choices=['eye', 'face', 'opticalflow'],
                     help='Which method to choose for processing the data')
parser.add_argument('ExperimentData',
                     help='The directory that holds all the data recorded at the\
                           N-Back experiment')
parser.add_argument('--ground-truth', '-gt',
                     choices=['n', 'score'],
                     default='n',
                     help='What should be used as the ground truth label. The n\
                           difficulty or the score the participant achieved')
parser.add_argument('--output', '-o',
                     default='.',
                     help='Directory to store the output in')
parser.add_argument('--window', '-w',
                     default=60,
                     help='The window size for a single input in frames')
parser.add_argument('--crop', '-c',
                     default=64,
                     help='The crop size for the frames')
parser.add_argument('--subsample', '-s',
                     default=1,
                     help='How many frames to subsample the time series by')
arguments = parser.parse_args()

# Argument Processing
method        = arguments.ProcessingMethod
exp_data_path = os.path.abspath(arguments.ExperimentData)
ground_truth  = arguments.ground_truth
output_path   = os.path.abspath(arguments.output)
windowsize    = int(arguments.window)
cropsize      = int(arguments.crop)
subsample     = int(arguments.subsample)


# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isdir(exp_data_path)
assert os.path.exists(output_path) and os.path.isdir(output_path)

assert windowsize > 0
assert cropsize > 0
assert subsample > 0

# retrieve the files from the experiment directory
exp_json, video_path, participant = dp.util.getExperimentInfo(exp_data_path)

assert os.path.exists(video_path) and os.path.isfile(video_path) \
    and os.path.splitext(video_path)[1] == '.mp4'
assert os.path.exists(exp_json) and os.path.isfile(exp_json) \
    and os.path.splitext(exp_json)[1] == '.json'


print('')
print('Processing Method: {}'.format(method))
print('Experiment Data: {}'.format(exp_json))
print('Video: {}'.format(video_path))
print('Ground Truth: {}'.format(ground_truth))
print('Output Path: {}'.format(output_path))
print('Windowsize: {}'.format(windowsize))
print('Cropsize: {}'.format(cropsize))
print('Subsampling: {}'.format(subsample))
print('Participant: {}'.format(participant))
print('')

# create the data processing objects
exp_data      = dp.ExperimentData(exp_json)
video_handler = dp.VideoHandler(video_path)

# set up the data handler according to the method
if(method in ['face', 'eye']):
    data_handler  = dp.DataHandler((windowsize, cropsize, cropsize),subsample)
# optical flow
else:
    data_handler  = dp.DataHandler((windowsize, cropsize, cropsize, 3), subsample)


# choose the right processing function accoding to the method
method_functions = {
    'face': video_handler.get_frames,
    'eye': video_handler.get_eye_frames,
    'opticalflow': video_handler.get_optical_flow_frames
}
process_frames = method_functions[method]

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
        frames = process_frames(trial['start'], trial['end'], cropsize)
        # set the ground truth
        gt = n if ground_truth == 'n' else trial['score']
        # store the away in the data handler with the correct label
        data_handler.add_frames(frames, gt)

print("Writing data to disk...")
# write the data to disk
data_path, labels_path = data_handler.write(output_path, participant, suffix=ground_truth)
print("  {}".format(data_path))
print("  {}".format(labels_path))
