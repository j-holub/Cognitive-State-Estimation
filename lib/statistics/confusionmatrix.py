import keras.models
import numpy as np

import sklearn.metrics


def confusion_mat(model: keras.Model, evaluation_data: np.ndarray, evaluation_labels: np.ndarray):
    # let the model predict the output for the data
    pred = model.predict(evaluation_data)
    # compute the confusion matrix
    mat = sklearn.metrics.confusion_matrix(evaluation_labels.argmax(axis=1), pred.argmax(axis=1))

    return mat/mat.sum(axis=1)
