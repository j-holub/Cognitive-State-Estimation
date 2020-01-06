""" Computes the confusion matrix from a trained model file and the validation
data and labels. The output is a numpy .npy file.

...

Arguments
    ResultsBaseDir (str):
        Directory containing the model file, the validation data and the validation
        labels
    --output, -o (optional, str):
        Directory where the output should be saved to
        default: '.'
"""

import argparse
import os

import numpy as np
import keras.models
import sklearn.metrics

from lib.statistics import confusion_mat

parser = argparse.ArgumentParser()
parser.add_argument('ResultsBaseDir',
                    help='Directory that contains a directory for every participant\
                          with the model file, the validation data and the validation\
                          laels')
parser.add_argument('--output', '-o',
                    help='Output directory',
                    default='.')
args = parser.parse_args()


assert os.path.exists(args.ResultsBaseDir) and \
       os.path.isdir(args.ResultsBaseDir)

assert os.path.exists(args.output) and \
       os.path.isdir(args.output)


dir = os.path.abspath(args.ResultsBaseDir)
out = os.path.abspath(args.output)

# gather all the participant directories
p_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]


# base prediction data
pred = np.zeros([1,5])
# base evaluation labels data
evaluation_labels = np.asarray([])

# iterate over every directory in the provided directory
for p in p_dirs:
    print(p)

    # path pointing to the currently processed directory
    p_dir = os.path.join(dir, p)
    # directory contents
    content = os.listdir(p_dir)

    # get the keras mode, validation data and validation labels from the directory
    model = keras.models.load_model(os.path.join(p_dir, [f for f in content if 'model' in f][0]))
    data = np.load(os.path.join(p_dir, [f for f in content if 'data' in f][0]))
    label = np.load(os.path.join(p_dir,[f for f in content if 'label' in f][0]))

    # have to model compute predictions for all the data
    for d in data:
        pred = np.concatenate((pred, model.predict(np.expand_dims(d,axis=0))), axis=0)

    # append the validation labels
    evaluation_labels = np.concatenate((evaluation_labels, label))

# compute the confusion matrix
conf_mat = sklearn.metrics.confusion_matrix(evaluation_labels, pred[1:,...].argmax(axis=1)+1, labels=range(1,6))
conf_mat = conf_mat/conf_mat.sum(axis=1, keepdims=True)


np.save(os.path.join(out,'confmat.npy'), conf_mat)

print('Confusion Matrix saved to {}...'.format(os.path.join(out, 'confmat.npy')))
