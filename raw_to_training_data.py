import argparse
import json
import os

import numpy as np

import lib.dataprocessing as dp

parser = argparse.ArgumentParser()
parser.add_argument('RawDataDir',
                     help='Directory where the frame .npy files and the ground\
                     truth .json files are stored')
parser.add_argument('--single-person-balanced', '-spb',
                     action='store_true',
                     help='If this flag is set, one balanced dataset with training and \
                           validation data for this single person will be created')
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


# data to be used in cross participant training
if not arguments.single_person_balanced:

    # initialize the data handler
    data_handler = dp.DataHandler((windowsize, *shape), subsample)

    # iterate over all the data
    for frames, gt in zip(frame_files, gt_files):
        # make sure the two files belong to each other
        assert os.path.splitext(os.path.basename(frames))[0] == os.path.splitext(os.path.basename(gt))[0]

        # open the gt file
        with open(gt, 'r') as gt_file:
            gt_data = json.load(gt_file)
            # load the frames
            frame_data = np.load(frames)
            # add the frames to the datahandler
            data_handler.add_frames(frame_data, gt_data[ground_truth])


    print("Writing data to disk...")
    # write the data to disk
    data_path, labels_path = data_handler.write(output_path, participant, suffix=ground_truth)
    print("  {}".format(data_path))
    print("  {}".format(labels_path))


# ----------------- 

# data to be used in single person training
else:
    # seperate datahandler for training and validation data
    train_data_handler = dp.DataHandler((windowsize, *shape), subsample)
    valid_data_handler = dp.DataHandler((windowsize, *shape), subsample)

    for i, (frames, gt) in enumerate(zip(frame_files, gt_files)):
        # make sure the two files belong to each other
        assert os.path.splitext(os.path.basename(frames))[0] == os.path.splitext(os.path.basename(gt))[0]

        # open the gt file
        with open(gt, 'r') as gt_file:
            gt_data = json.load(gt_file)
            # load the frames
            frame_data = np.load(frames)

            # validation set
            # last one of each difficulty is used for validation
            if (i+1)%5==0:
                valid_data_handler.add_frames(frame_data, gt_data[ground_truth])
            # training set
            # trials 0-3 are used for training
            else:
                train_data_handler.add_frames(frame_data, gt_data[ground_truth])


    print("Writing data to disk...")
    # write the data to disk
    data_path, labels_path = train_data_handler.write(output_path, participant, suffix='{}_train'.format(ground_truth))
    print("  {}".format(data_path))
    print("  {}".format(labels_path))
    data_path, labels_path = valid_data_handler.write(output_path, participant, suffix='{}_validation'.format(ground_truth))
    print("  {}".format(data_path))
    print("  {}".format(labels_path))
