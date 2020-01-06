""" This script computes the predictions for the data recorded while the
participants were watching a lecture video, to see how their cognitive load
changes throughout the course of the lecture video

It outputs the predictions as numpy .npy file

...

Arguments:
    Frames (str):
        path to file containing all the training samples for the lecture video part
    Model (str):
        path to a saved keras model file that was trained beforehand
    --output, -o (optional, str):
        path where the output file should be saved to
    --participant, -p (optional, str):
        can be used to add the any string to the output file. It is meant for the
        participant ID
"""

import argparse
import os

import keras.models
import numpy as np

import lib.deeplearning as deepl

parser = argparse.ArgumentParser()
parser.add_argument('Frames',
                     help='The frames coupled in chunks of windowsize')
parser.add_argument('Model',
                     help='Trained model file')
parser.add_argument('--output', '-o',
                     default='.',
                     help='Output directory')
parser.add_argument('--participant', '-p',
                     help='participant indentifier that should be added to\
                           the output file')
args = parser.parse_args()


assert os.path.exists(args.Frames) \
    and os.path.isfile(args.Frames) \
    and os.path.splitext(args.Frames)[1] == '.npy'

assert os.path.exists(args.Model) \
    and os.path.isfile(args.Model) \
    and os.path.splitext(args.Model)[1] == '.h5'


# load the coupled frames
print('Loading frames...')
frames = np.load(args.Frames)
print('Done')


print('Loading model...')
model = keras.models.load_model(args.Model)
print('Done')


predictions = np.ndarray((1,5))

print('Computing predictions...')
for window in frames:
    # predictions = model.predict(frames)
    pred = model.predict(np.expand_dims(window,axis=0))
    predictions = np.concatenate((predictions, pred),axis=0)
print('Done')

# output file
out = os.path.join(args.output,
    'lecture_video_predictions.npy' \
        if not args.participant \
        else 'p{}_lecture_video_predictions.npy'.format(args.participant))

print('Saving predictions to {}..'.format(out))
np.save(out, predictions[1:,...])
