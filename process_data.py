import argparse
import os

from lib.dataprocessing.experimentdata import ExperimentData


# Argument Parsing

parser = argparse.ArgumentParser()
parser.add_argument('ExperimentData',
                     help='The JSON file produced by the JsPsych N-Back Experiment')
arguments = parser.parse_args()

# path to the JSON file created by the JsPsych N-Back Experiment
exp_data_path = os.path.abspath(arguments.ExperimentData)


# Assertion Checks
assert os.path.exists(exp_data_path) and os.path.isfile(exp_data_path)

exp_data = ExperimentData(exp_data_path)
