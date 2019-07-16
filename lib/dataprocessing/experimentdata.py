import json
import os

class ExperimentData:
    """
    Class representing an object that stores and processes all the data gathered
    from the JsPsych N-Back experiment

    It loads the data, processes it and groups it

    ...

    """




    def __init__(self, expdata: str):
        """
        Parameters:
            expdata (str): path to the JSON file created by the JsPsych N-Back
                Experiment
        """

        # open the experiment data file, that stores
        # all the information
        with open(expdata, 'r') as json_file:
            # load the json data
            data = json.load(json_file)

            # extract the n-back data
            # 1-back
            one_back   = [trial for trial in data if trial['test_part'] == 'n-back' and trial['n'] == 1]
            # 2-back
            two_back   = [trial for trial in data if trial['test_part'] == 'n-back' and trial['n'] == 2]
            # 3-back
            three_back = [trial for trial in data if trial['test_part'] == 'n-back' and trial['n'] == 3]
            # 4-back
            four_back  = [trial for trial in data if trial['test_part'] == 'n-back' and trial['n'] == 4]
            # 5-back
            five_back  = [trial for trial in data if trial['test_part'] == 'n-back' and trial['n'] == 5]


            self.__n_back_data = {}
            self.__n_back_data[1] = self.__group_n_back_trials(one_back)
            self.__n_back_data[2] = self.__group_n_back_trials(two_back)
            self.__n_back_data[3] = self.__group_n_back_trials(three_back)
            self.__n_back_data[4] = self.__group_n_back_trials(four_back)
            self.__n_back_data[5] = self.__group_n_back_trials(five_back)




    def __node_id(self, node_id: str, index: int):
        """Extracts a single id of JsPsych's 'internal_node_id' property

        Given the 'internal_node_id' string of JsPsych it returns the id at
        index (0 based)

        Parameters:
            node_id (str): string retrieved from the 'internal_node_id' property
                retrieved from a JsPsych stimuli JSON object
            index (int): the index of which id part should be returned
                (0 based)

        Returns:
            float: the node id for the given index
        """

        return float(node_id.split('-')[index])



    def __group_n_back_trials(self, trials: list):
        """Groups a list of various stimuli into lists of trials

        Given a list if stimuli belonging to differnet trials this method will
        create a list for each trial containing the stimuli and return a list
        of those lists

        Parameters:
            trials (list): list of trial JSON objects produced by JsPsych

        Returns:
            list: a list of grouped trials, one list for each trials
        """

        # get the secondary internal node id to identify the different trials
        node_ids = set(map(lambda trial:self.__node_id(trial['internal_node_id'], 1), trials))
        # group the trials according to the secondary node id such that each list
        # is one trial
        grouped_trials = [
            [trial for trial in trials if self.__node_id(trial['internal_node_id'], 1) == id]
        for id in node_ids]

        return grouped_trials
