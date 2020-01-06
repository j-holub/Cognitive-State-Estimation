import keras.models
import numpy as np

import sklearn.metrics


def confusion_mat(model: keras.Model, evaluation_data: np.ndarray, evaluation_labels: np.ndarray):
    """ Computes a confusion matrix given a model, evaluation data and ground
        truth labels

        Computes predictions for every give data sample and creates a confusion matrix
        using the provided ground truth data

        model (keras.Model):
            a trained keras model
        evaluation_data:
            data samples that were not used during training
        evaluation_labels:
            ground truth labels for the evaluation data

        Returns:
            np.ndarray: the confusion matrix normalised between 0 and 1
    """
    pred = np.zeros([1,5])
    # let the model predict the output for the data
    for d in evaluation_data:
        pred = np.concatenate((pred, model.predict(np.expand_dims(d,axis=0))), axis=0)

    # print(pred[1:,...])
    # print(evaluation_labels)
    # compute the confusion matrix
    mat = sklearn.metrics.confusion_matrix(evaluation_labels, pred[1:,...].argmax(axis=1)+1)

    return mat/mat.sum(axis=1, keepdims=True)
