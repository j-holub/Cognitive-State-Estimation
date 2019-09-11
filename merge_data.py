"""Merges the data and labels .npy files from all participants into one file

This scripts serves the purpose to merge the data of all the participants into
one big .npy file that can then be input to a neural network

---------------

Parameters:
    Data (str):
        Directory containing all the different .npy files of the participants
    output (str, optional):
        Output directory. Same as Data if not specified
"""

import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('Data',
                     help='Directory containing all the processed .npy files')
parser.add_argument('--output', '-o',
                     help='Output directory for the combined .npy files. Same as \
                           Data if not specified')
arguments = parser.parse_args()

dir     = os.path.abspath(arguments.Data)
out_dir = os.path.abspath(arguments.output) \
            if arguments.output \
            else os.path.abspath(arguments.Data)


assert os.path.exists(dir) and os.path.isdir(dir)
assert os.path.exists(out_dir) and os.path.isdir(out_dir)



data_files = sorted([file for file in os.listdir(dir) if 'data' in file])
label_files = sorted([file for file in os.listdir(dir) if 'label' in file])

assert len(data_files) == len(label_files)

print('')
print('Data directory: {}'.format(dir))
print('Output directory: {}'.format(out_dir))
print('Number of participant data found: {}'.format(len(data_files)))
print('')

# load the first label file to get the shape
data_shape = np.load(os.path.join(dir, data_files[0])).shape

# basic name pattern for the output file
outfile_name = 'all_{}'.format(data_files[0][4:-9])

# labels
all_labels = np.ndarray(data_shape[0]*len(data_files))

# memory map for the data
all_data = np.memmap('{}.memmap'.format(outfile_name), dtype='uint8', mode='w+', \
            shape=(data_shape[0]*len(data_files), *data_shape[1:]))

print('Merging files...')
# transfer all the data to the memory map
for i, (data_file, labels_file) in (enumerate(zip(data_files, label_files))):
    print('  ({}, {})'.format(data_file, labels_file))
    # check that the 2 files belong to the same participant by comparing
    # the first 3 letters namely pXX
    assert data_file[:3] == labels_file[:3]
    data   = np.load(os.path.join(dir,data_file))
    labels = np.load(os.path.join(dir,labels_file))

    all_data[i*data_shape[0]:i*data_shape[0]+data_shape[0],...] = data
    all_labels[i*data_shape[0]:i*data_shape[0]+data_shape[0]] = labels
print('')

print('Writing data to disc...')
# save the data
np.save(os.path.join(out_dir, '{}_data.npy'.format(outfile_name)), all_data)
np.save(os.path.join(out_dir, '{}_labels.npy'.format(outfile_name)), all_labels)
print("  {}".format(os.path.join(out_dir, '{}_data.npy'.format(outfile_name))))
print("  {}".format(os.path.join(out_dir, '{}_labels.npy'.format(outfile_name))))

# remove the memory map
os.remove('{}.memmap'.format(outfile_name))
