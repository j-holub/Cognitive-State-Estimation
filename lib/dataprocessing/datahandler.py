import math
import os

import numpy as np

class DataHandler:
    """
    Class the handles the processed data. It stores the data and splits the frames
    into chunks of a windowsize

    It can write the data to disk

    Methods:
        add_frames(frames, label)
            adds frames to the data, it groups them into chunks of windowsize
            and adds a label for each chunk
        get_data()
            returns data and labels
        write(path, name)
            writes the data and labels into an output directory with a name
            attached
    """


    def __init__(self, shape: tuple, subsample: int = 1):
        """
        Parameters:
            shape (tuple): the shape of the data image data without the
                the axis that corresponds to the amount of data
            subsample (int): how many frames to subsample the time series by
                default: 1 (no subsampling at all)
        """
        # subsample rate
        self.__subsample = subsample

        # the numpy array to store all the frames in
        self.__data   = np.zeros([1, *shape], dtype=np.uint8)
        # the numpy array to hold the labels
        self.__labels = np.zeros(1, dtype=np.uint8)




    def add_frames(self, frames: np, ground_truth: int):
        """Adds frames in chunks of the windowsize to the data with the label

        The frames given are split into chunks of windowsize, with the last frames
        possiblity omitted, if they don't fit

        For each chunks one label is added to the label array of the class

        Parameters:
            frames (ndarray):
                numpy array of frames
            ground_truth (int):
                ground truth value for the frames
        """

        # get the windowsize
        windowsize = self.__data.shape[1]

        for x in range(0, self.__subsample):
            # subsample the frames using numpy magic
            subsampled_frames = frames[x::self.__subsample]
            # see how many frames we need to drop in the end to get chunks
            # of the windowsize
            diff = subsampled_frames.shape[0] % windowsize
            # get the number of chunks
            chunks = math.floor(subsampled_frames.shape[0] / windowsize)
            # split the frames into segments of windowsize
            segments = np.split(subsampled_frames[:-diff,...], chunks, axis=0) \
                if diff > 0 \
                else np.split(subsampled_frames, chunks, axis=0)

            # add the segments to the data
            for segment in segments:
                self.__data   = np.concatenate((self.__data, np.expand_dims(segment, axis=0)), axis=0)
                self.__labels = np.append(self.__labels, np.expand_dims(label, axis=0))

        assert self.__data.shape[0] == self.__labels.shape[0]



    def get_data(self):
        """returns data and labels

        Returns:
            ndarray, ndarray: numpy array with the data and numpy array with the
                labels
        """

        return self.__data[1:,...], self.__labels[1:]



    def write(self, path: str, name: str, suffix: str = ''):
        """Writes the stored data to a file

        This method writes all the data stored so far to the output file. It
        should be called after all the data is collected

        Parameters:
            path (str):
                path to the directory where the output should be stored in
            name (str):
                name the files should have, for example the participant number
            suffix (str, optional):
                some arbtrary string that is added to the file name

        Returns:
            str, str: output path to the data and the labels file
        """

        # create the filename
        name_str = '{}_{}@{}_{}x{}{}'.format(
            name,
            self.__data.shape[1],
            self.__subsample,
            self.__data.shape[2],
            self.__data.shape[2],
            '_{}'.format(suffix) if suffix else suffix
        )

        # path to the numpy files
        data_path   = os.path.join(path, '{}_data.npy'.format(name_str))
        labels_path = os.path.join(path, '{}_labels.npy'.format(name_str))

        # write the data without the very first zero entry
        np.save(data_path, self.__data[1:,...])
        np.save(labels_path, self.__labels[1:])

        return data_path, labels_path
