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

    def __init__(self, features_file: str, labels_file: str, val_split: float):
        """
        Parameters:
            features_file (str):
                path to the .npy file containing the features
            labels_file (str):
                path to the .npy file containing the labels
            val_split (float):
                ratio to split the data into train and test set. Must be in the
                interval (0,1)
        """

        assert val_split>0 and val_split<1

        # load the data
        self.__data = np.load(features_file, mmap_mode='r+')
        # load the labels and transform them to 1 hot
        self.__labels = np.load(labels_file)
        self.__labels = keras.utils.to_categorical(self.__labels)[...,1:]

        assert self.__data.shape[0] == self.__labels.shape[0]

        # transform the data if needed
        # if the dimension is (amount, window, crop crop) transform
        # it to (amount, window, crop, crop, 1)
        if(len(self.__data.shape)==4):
            self.__data = np.reshape(self.__data, (*self.__data.shape, 1))

        # compute the index to divide the data into training and validation set
        self.__split_index = int(self.__data.shape[0] * val_split)

        # normalize the data
        self.__data = (self.__data - np.mean(self.__data)) / np.std(self.__data)


    def train_data(self):
        """Returns the train dataset

        Returns:
            (np.ndarray, np.ndarray): train features and train labels as numpy
                arrays of the according dimensions

        """

        return self.__data[:self.__split_index], self.__labels[:self.__split_index]


    def test_data(self):
        """Returns the test dataset

        Returns:
            (np.ndarray, np.ndarray): test features and test labels as numpy
                arrays of the according dimensions

        """

        return self.__data[self.__split_index:], self.__labels[self.__split_index:]
