import numpy as np

from ..dataprocessing import ExperimentData

class Statistic:
    """Class that stores the data from the different experiments and offers various
       methods to compute statistics and metrics on that data

       It stores the data in a big numpy array matrix, one row per subject, which
       makes the computing of the various metrics very efficient since most of them
       are implemented in NumPy anyways

       ...

       Methods:
            add_subject_data(data_csv_path)
                extracs the relevant data from the csv file generated by JsPsych
                and stores it
            average_scores_all_subjects()
                computes the average score for n=1 to n=5 for each subjects and
                returns it as a matrix (np.ndarray)
            global_average_scores()
                computes the averages of the average_scores for each n level and
                returns them as a 5 element vector
            global_std_deviation()
                computes the standard deviations of the average_scores for each n
                level and returns them as a 5 element vector
            global_score_count()
                returns a 5x10 matrix as a numpy array to shows for which difficulty
                levels (1-5), how often the possible scores (1-10) were achieved
    """


    def __init__(self):

        # set the data to False intially
        # will be overwritten by the first data that is added
        # rows: participant
        # cols: trial in order 1 1 1 1 1 2 2 2 ... 4 4 5 5 5 5 5
        self.__data = np.array([False])



    def add_subject_data(self, data_csv_path: str):
        """Reads a csv file form the n back experiment and stores the scores

        The scores are scored in a matrix, one row per participant with each
        column representing a trial

        Parameters:
            data_csv_path (str): path to the csv file generated by JsPsych
        """

        # load the subjects data
        exp_data = ExperimentData(data_csv_path)

        # data array as a matrix
        data = np.zeros([1,25], dtype=float)

        # iterate over the n steps
        for n in range(1,6):
            # iterate over every trial for that n step
            for i, trial in enumerate(exp_data.get_trials(n)):
                # write the score to the correct position
                data[0,((n-1)*5)+i] = trial['score']

        # add the data to the existing data
        if(self.__data.any()):
            self.__data = np.concatenate((self.__data, data), axis=0)
        else:
            self.__data = data



    def average_scores_all_subjects(self):
        """Computes the average for each subject over each n level

            Per subject 5 average scores will be computed, n=1 - n=5

            Returns:
                np.ndarray: matrix with a row per subject and 5 entries per row,
                    1 for each n step
        """

        # number of participants
        p_num = self.__data.shape[0]
        # output matrix
        out = np.zeros([p_num, 1])

        for n in range(1,6):
            mean = np.mean(self.__data[...,(n-1)*5:(n-1)*5+5], axis=1)
            out = np.concatenate((out, np.expand_dims(mean, axis=1)), axis=1)

        return out[...,1:]



    def global_average_scores(self):
        """Computes the average for each n across all subjects

            Returns:
                np.ndarray: 5x1 matrix with the average scores for n=1 to n=5
        """

        return np.mean(self.average_scores_all_subjects(), axis=0)


    def global_std_deviation(self):
        """Computes the standard deviation for each n across all subjects

            Returns:
                np.ndarray: 5x1 matrix with the standard deviation scores for n=1 to n=5
        """

        return np.std(self.average_scores_all_subjects(), axis=0)


    def global_score_count(self):
        """For each n (1-5) and possible score (1-10) how often they occured

            Returns:
                np.ndarray: 5x10 matrix with columns as score and rows as n

        """
        score_count_mat = np.ndarray((5,10))

        for n in range(0,5):
            score_count = [
                np.count_nonzero(self.__data[...,n*5:n*5+5] == score)
                for score
                in np.arange(10)+1
            ]

            score_count_mat[n,...] = score_count

        return score_count_mat
