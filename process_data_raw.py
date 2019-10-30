import argparse
import json
import os

import numpy as np

import lib.dataprocessing as dp


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('ProcessingMethod',
                     choices=['eye', 'face'],
                     help='Which method to choose for processing the data')
parser.add_argument('ExperimentData',
                     help='The directory that holds all the data recorded at the\
                           N-Back experiment')
parser.add_argument('--output', '-o',
                     default='.',
                     help='Directory to store the output in')
parser.add_argument('--crop', '-c',
                     default=64,
                     help='The crop size for the frames')
parser.add_argument('--lecture-video', '-lv',
                     default=False,
                     action='store_true',
                     help='If set the frames for the lecture video are extracted')
arguments = parser.parse_args()


# Argument Processing
method        = arguments.ProcessingMethod
exp_data_path = os.path.abspath(arguments.ExperimentData)
output_path   = os.path.abspath(arguments.output)
cropsize      = int(arguments.crop)
lecture_video = arguments.lecture_video


# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isdir(exp_data_path)
assert os.path.exists(output_path) and os.path.isdir(output_path)

assert cropsize > 0

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
print('Output Path: {}'.format(output_path))
print('Cropsize: {}'.format(cropsize))
print('Participant: {}'.format(participant))
print('Video Part: {}'.format('Lecture Video' if lecture_video else 'N-Back'))
print('')

# create the data processing objects
exp_data      = dp.ExperimentData(exp_json)
video_handler = dp.VideoHandler(video_path)

# choose the right processing function accoding to the method
method_functions = {
    'face': video_handler.get_frames,
    'eye': video_handler.get_eye_frames,
}
process_frames = method_functions[method]

# create the output directory for this participant
os.makedirs(os.path.join(output_path, participant), exist_ok=True)

if not lecture_video:

    print('Processing N-levels...')
    # go over every difficulty level
    for n in range(1,6):
        print('N={}'.format(n))
        # get the trial timestampts and score
        trials = exp_data.get_trials(n)

        # iterate over every trial for this difficulty
        for i, trial in enumerate(trials):
            print('  Trial {}'.format(i+1))
            # get the frames
            frames = process_frames(trial['start'], trial['end'], cropsize)

            # format the output path and filename
            out = os.path.join(output_path, participant, '{}-{}-{}'.format(n,i, method))
            # save the frames
            np.save('{}.npy'.format(out), frames)
            # save the gt data as a json file
            with open('{}.json'.format(out), 'w') as json_file:
                json.dump({
                    'n': n,
                    'score': trial['score']
                }, json_file)

else:

    print('Processing Lecture Video')

    # get the timestamps for the lecture video
    start, end = exp_data.get_video_lecture_timestamps()
    # process the frames
    frames = process_frames(start, end, cropsize)
    # format the output path and filename
    out = os.path.join(output_path, participant, '{}_lecture_video'.format(method))
    np.save('{}.npy'.format(out), frames)

print('Output written to {}'.format(os.path.join(output_path, participant)))
