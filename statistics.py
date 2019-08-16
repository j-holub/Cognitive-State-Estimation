import argparse
import json
import os

import lib.dataprocessing as dp


parser = argparse.ArgumentParser()
parser.add_argument('ExperimentData', nargs='*',
                     help='All the experiment json files that should be included')
parser.add_argument('--output', '-o', default='statistics.json',
                     help='The output JSON file to store the statistics')
arguments = parser.parse_args()

# get the absolute paths to the csv files
exp_data_files = [os.path.abspath(file) for file in arguments.ExperimentData]
out_file       = os.path.abspath(arguments.output)


assert all([
    os.path.exists(file) and
    os.path.isfile(file) and
    os.path.splitext(file)[1] == '.json'
    for file in exp_data_files
])

assert os.path.splitext(out_file)[1] == '.json'

# statistic object
stat = dp.Statistic()

# add all the data from the experiments
for data in exp_data_files:
    stat.add_subject_data(data)

# get the relevant metrics
average_subject_scores = stat.average_scores_all_subjects()
global_average_scores  = stat.global_average_scores()

# create the statistics object
statistics = {}

# average subject scores
statistics['average_scores'] = []
for i in range(average_subject_scores.shape[0]):
    statistics['average_scores'].append(average_subject_scores[i,...].tolist())
# global average
statistics['global_average'] = global_average_scores.tolist()

# write the output
with open(out_file, 'w') as out:
    json.dump(statistics, out)
    print('Saved output to {}'.format(out_file))
