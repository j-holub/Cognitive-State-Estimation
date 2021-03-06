""" This script takes the .npy files extraced by the process_data_raw.py script
and couples them into files for the training and validation of a model. The script
works on the data of a single participant and has to be evoked multiple times if you
want to process the data of multiple participants

This scripts couples the frames into chunks of a certain windowsize (default is 60)
and can subsample them if needed.

The data can be grouped as a balanced dataset for a single person, where 4 trials
of each difficulty level are used for the training set and 1 for the validation
set. This provides a perfectly balanced dataset to train a model for a single
participant
The data can also be grouped into a single data and a single ground truth file
to combine it with the data of other participants.

...

Arguments:
    RawDataDir
        The directory containing all the output files (.npy and .json) generated
        by the process_data_raw.py script
    --single-person-balanced, -spb (optional, flag)
        If set, the data is split into a training and validation set for this
        single person. The out will be 4 files, training plus validation data and
        training plus validation labels
    --two-classm -tc (optional, flag)
        If set, classes 1 and 2 will be merged into one class, 4 and 5 will be
        merged and 3 will be omitted. This creates a two class dataset,
        corresponding to low and high cognitive load
    --ground-truth, -gt (optical, str)
        Must be one of 'n' or 'score'. Denotes which ground truth measure should
        be used, either the difficulty n of the n-back trial or the score value
        achieved on the trial
        default: 'n'
    --window, -w (optional, int)
        The window sized that should be used
        default: 60
    --subsample, -ss (optional, int)
        Subsampling rate that should be used. If set to x>1, every x-th frame
        will be used for a window
        default: 1
    --output, -o (optional, str)
        The output directory
        default: '.'
"""


import argparse
import json
import os

import numpy as np

from lib.dataprocessing import DataHandler

parser = argparse.ArgumentParser()
parser.add_argument('RawDataDir',
                     help='Directory where the frame .npy files and the ground\
                     truth .json files are stored')
parser.add_argument('--single-person-balanced', '-spb',
                     action='store_true',
                     help='If this flag is set, one balanced dataset with training and \
                           validation data for this single person will be created')
parser.add_argument('--two-class', '-tc',
                     action='store_true',
                     help='If set, classes 1,2  and 4,5 will be merged into one \
                           class and class 3 will be omitted. This results in a \
                           two class dataset')
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
parser.add_argument('--subsample', '-ss',
                     default=1,
                     help='How many frames to subsample the time series by')
arguments = parser.parse_args()


# Argument Processing
exp_data_path = os.path.abspath(arguments.RawDataDir)
ground_truth  = arguments.ground_truth
output_path   = os.path.abspath(arguments.output)
windowsize    = int(arguments.window)
subsample     = int(arguments.subsample)
spb           = arguments.single_person_balanced
twoclass      = arguments.two_class

# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isdir(exp_data_path)
assert os.path.exists(output_path) and os.path.isdir(output_path)

assert windowsize > 0
assert subsample > 0

participant = os.path.basename(exp_data_path)

print('')
print('Raw Data: {}'.format(exp_data_path))
print('Ground Truth: {}'.format(ground_truth))
print('Output Path: {}'.format(output_path))
print('Windowsize: {}'.format(windowsize))
print('Subsampling: {}'.format(subsample))
print('Participant: {}'.format(participant))
print('Single Person Balanced: {}'.format(spb))
print('Two Class: {}'.format(twoclass))
print('')

# list all the files containing the frames
frame_files = sorted([os.path.join(exp_data_path, file)
                for file
                in os.listdir(exp_data_path)
                if os.path.splitext(os.path.join(exp_data_path, file))[1] == '.npy'
              ])

# list all the files containing the ground truth values
gt_files    = sorted([os.path.join(exp_data_path, file)
                for file
                in os.listdir(exp_data_path)
                if os.path.splitext(os.path.join(exp_data_path, file))[1] == '.json'
              ])

# get the shape of a single frame
shape = np.load(frame_files[0]).shape[1:]

# suffix to add to the output files
suffix = ground_truth

# remove classes 2-3 if twoclass is set
if twoclass:
    frame_files = [file
                    for file
                    in frame_files
                    if os.path.basename(file)[0] != '3'
                    # or os.path.basename(file)[0] == '5'
                  ]
    gt_files = [file
                    for file
                    in gt_files
                    if os.path.basename(file)[0] != '3'
                    # or os.path.basename(file)[0] == '5'
                  ]
    suffix = suffix + '_twoclass'


# data to be used in cross participant training
if not spb:

    # initialize the data handler
    data_handler = DataHandler((windowsize, *shape, 1), subsample)

    # iterate over all the data
    for frames, gt in zip(frame_files, gt_files):
        # make sure the two files belong to each other
        assert os.path.splitext(os.path.basename(frames))[0] == os.path.splitext(os.path.basename(gt))[0]

        # open the gt file
        with open(gt, 'r') as gt_file:
            gt_data = json.load(gt_file)
            # load the frames
            frame_data = np.load(frames)

            # set the ground truth accordingly
            if ground_truth == 'n' and twoclass:
                # merge 1,2 and 4,5 into one class
                gt = 1 if gt_data[ground_truth] in [1,2] else 2
            else:
                gt = gt_data[ground_truth]

            # add the frames to the datahandler
            data_handler.add_frames(frame_data, gt)


    print("Writing data to disk...")
    # write the data to disk
    data_path, labels_path = data_handler.write(output_path, participant, suffix=suffix)
    print("  {}".format(data_path))
    print("  {}".format(labels_path))


# -----------------

# data to be used in single person training
else:
    # seperate datahandler for training and validation data
    train_data_handler = DataHandler((windowsize, *shape, 1), subsample)
    valid_data_handler = DataHandler((windowsize, *shape, 1), subsample)

    for i, (frames, gt) in enumerate(zip(frame_files, gt_files)):
        # make sure the two files belong to each other
        assert os.path.splitext(os.path.basename(frames))[0] == os.path.splitext(os.path.basename(gt))[0]

        # open the gt file
        with open(gt, 'r') as gt_file:
            gt_data = json.load(gt_file)
            # load the frames
            frame_data = np.load(frames)

            # load the ground truth data from the json file
            # if the ground truth metric is n and the two class option
            # is set, class 5 should be labeled with 2 for the one hot
            # encoding returned by keras.utils.to_categorical
            if ground_truth == 'n' and twoclass:
                # merge 1,2 and 4,5 into one class
                gt = 1 if gt_data[ground_truth] in [1,2] else 2
            else:
                gt = gt_data[ground_truth]


            # validation set
            # last one of each difficulty is used for validation
            if (i+1)%5==0:
                valid_data_handler.add_frames(frame_data, gt)
            # training set
            # trials 0-3 are used for training
            else:
                train_data_handler.add_frames(frame_data, gt)


    print("Writing data to disk...")
    # write the data to disk
    data_path, labels_path = train_data_handler.write(output_path, participant, suffix='{}_train'.format(suffix))
    print("  {}".format(data_path))
    print("  {}".format(labels_path))
    data_path, labels_path = valid_data_handler.write(output_path, participant, suffix='{}_validation'.format(suffix))
    print("  {}".format(data_path))
    print("  {}".format(labels_path))
