import datetime
import json
import os

class ExperimentData:
    """
    Class representing an object that stores and processes all the data gathered
    from the JsPsych N-Back experiment

    It loads the data, processes it and groups it

    ...

    Methods:
        get_trials(n)
            returns all trials with timestamps for a given n
        get_video_lecture_timestamps()
            returns a tuple with start and end timestamp for the video lecture
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

            # dictionary to hold the trials for a n difficulty
            self.__n_back_data = {}

            # group the trials by difficulty
            for n in range(1,6):
                # get lists for each difficulty
                trials = [trial for trial in data if trial['test_part'] == 'n-back' and trial['n'] == n]
                # group the stimuli into their respective trial
                self.__n_back_data[n] = self.__group_n_back_trials(trials)

            # video lecture data is always the last trial
            self.__video_lecture_data = data[-1]




    def get_trials(self, n: int):
        """ Gets the relevant data for all trials of difficulty level n

        Extracts the first and last relevant timestamp of each trial along with
        the difficulty level n

        Parameters:
            n (int): difficulty level / n step. Range 1-5

        Returns:
            list: list of trials of difficulty level n, with starting and ending
                timestamp
        """
        # list to store the trials information
        trial_data = []
        # retrieve the data
        trials = self.__n_back_data[n]

        for trial in trials:
            # starting timestamp is the first stimulus the subject is able to answer for
            timestamp_start = datetime.datetime.fromtimestamp(trial[n]['trial_start']/1000)
            # ending timestemp is the last stimulus
            timestamp_end   = datetime.datetime.fromtimestamp(trial[-1]['trial_end']/1000)

            # append the data to the trials list
            trial_data.append({
                'start': timestamp_start,
                'end': timestamp_end,
                'n': n
            })

        return trial_data



    def get_video_lecture_timestamps(self):
        """Gets the starting and ending timestamp of the video lecture

        Returns:
            :rtype: (datetime.datetime, datetime.datetime): tuple of starting and
                ending timestamp
        """

        start = datetime.datetime.fromtimestamp(self.__video_lecture_data['trial_start']/1000)
        end = datetime.datetime.fromtimestamp(self.__video_lecture_data['trial_end']/1000)

        return (start, end)



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
