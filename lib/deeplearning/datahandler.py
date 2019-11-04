import numpy as np
import keras.utils

class DataHandler:
    """Class to store and handle training data for neural networks

    It normalizes the data, shuffles it and splits it into a test and a training
    set. It also transforms the labels into one hot representation.

    Methods:
        train_data()
            returns features and labels for the train set
        test_data()
            returns features and labels for the test set
    """



    def __init__(self,
        train_feat_file: str,
        train_labels_file: str,
        valid_feat_file: str,
        valid_labels_file: str,
        regression: bool = False):
        """
        Parameters:
            train_feat_file (str):
                path to the .npy file containing the training features
            train_labels_file (str):
                path to the .npy file containing the training labels
            valid_feat_file (str):
                path to the .npy file containing the validation features
            valid_labels_file (str):
                path to the .npy file containing the validation labels
            regression (bool):
                if set to False labels will be converted to one hot encoding, if
                set to true, they will be left as they are
                default: False
        """

        # load the training data
        self.__train_data = np.load(train_feat_file, mmap_mode='r+')
        # load the training labels and transform them to 1 hot
        labels = np.load(train_labels_file)
        self.__train_labels = labels \
                                if regression \
                                else keras.utils.to_categorical(labels)[...,1:]

        assert self.__train_data.shape[0] == self.__train_labels.shape[0]

        # load the validation data
        self.__valid_data = np.load(valid_feat_file, mmap_mode='r+')
        # load the validation labels and transform them to 1 hot
        labels = np.load(valid_labels_file)
        self.__valid_labels = labels \
                                if regression \
                                else keras.utils.to_categorical(labels)[...,1:]

        assert self.__valid_data.shape[0] == self.__valid_labels.shape[0]

        # transform the data if needed
        # if the dimension is (amount, window, crop crop) transform
        # it to (amount, window, crop, crop, 1)
        if(len(self.__train_data.shape)==4):
            assert self.__train_data.shape[1:] == self.__valid_data.shape[1:]
            self.__train_data = np.reshape(self.__train_data, (*self.__train_data.shape, 1))
            self.__valid_data = np.reshape(self.__valid_data, (*self.__valid_data.shape, 1))



    def train_data(self):
        """Returns the train dataset

        Returns:
            (np.ndarray, np.ndarray): train features and train labels as numpy
                arrays of the according dimensions

        """

        return self.__train_data, self.__train_labels



    def test_data(self):
        """Returns the test dataset

        Returns:
            (np.ndarray, np.ndarray): test features and test labels as numpy
                arrays of the according dimensions

        """

        return self.__valid_data, self.__valid_labels
