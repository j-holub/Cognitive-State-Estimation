import keras.models
import numpy as np

import sklearn.metrics


def confusion_mat(model: keras.Model, evaluation_data: np.ndarray, evaluation_labels: np.ndarray):

    pred = np.zeros([1,5])
    # let the model predict the output for the data
    for d in evaluation_data:
        pred = np.concatenate((pred, model.predict(np.expand_dims(d,axis=0))), axis=0)

    # print(pred[1:,...])
    # print(evaluation_labels)
    # compute the confusion matrix
    mat = sklearn.metrics.confusion_matrix(evaluation_labels, pred[1:,...].argmax(axis=1)+1)

    return mat/mat.sum(axis=1, keepdims=True)
