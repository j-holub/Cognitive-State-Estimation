import argparse
import os

import lib.dataprocessing as dp


# Argument Parsing

parser = argparse.ArgumentParser()
parser.add_argument('ExperimentData',
                     help='The JSON file produced by the JsPsych N-Back Experiment')
arguments = parser.parse_args()

# path to the JSON file created by the JsPsych N-Back Experiment
exp_data_path = os.path.abspath(arguments.ExperimentData)
video_path    = os.path.abspath(arguments.Video)


# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isfile(exp_data_path)


exp_data = dp.ExperimentData(exp_data_path)
