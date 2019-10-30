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


print('Computing predictions...')
predictions = model.predict(frames)
print('Done')

# output file
out = os.path.join(args.output,
    'lecture_video_predictions.npy' \
        if not args.participant \
        else 'p{}_lecture_video_predictions.npy'.format(args.participant))

print('Saving predictions to {}..'.format(out))
np.save(out, predictions)
