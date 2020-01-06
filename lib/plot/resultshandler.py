import json
import os
import re

import numpy as np

class ResultsHandler:
    """ This class stores the results from training multiple models and gives access to
        different measures

        It loads output files produced by training modes on datasets using
        Keras. Different functions give acccess to metrics such as the accuracy
        or the loss

        add_result_file(path)
            loads a result file produced by training a model using Keras
        get_persons()
            returns the IDs for the different persons, who's result files were
            added using the add_result_file method
        get_test_acc()
            returns a dictionary with the maximum training accuracy for each
            participant
        get_val_acc()
            returns a dictionary with the maximum validation accuracy for each
            participant
        get_test_loss()
            returns a dictionary with the minimum training loss for each
            participant
        get_val_loss()
            returns a dictionary with the minimum validation loss for each
            participant
        get_metric(metric)
            metric must be one of 'accuracy', 'loss', 'val_accuracy' and 'val_loss'
            returns the maximum (accuracy) or minimum (loss) value for each participant
    """


    def __init__(self):
        # empty dictionary for the results
        self.__results = {}



    def add_result_file(self, path: str):
        # read the file
        with open(path, 'r') as his:
            # load the json data
            data = json.load(his)
            # get the person
            p = self.__get_person_name(os.path.basename(path))
            # add it to the results
            self.__results[p] = data



    def __get_person_name(self, filename: str):
        """ gets the IDs for a participant from the filename

            filename (String):
                filename of the result file

            Returns:
                String: participant ID
        """

        match = re.search('p\d\d', filename)
        return match.group()

    def get_persons(self):
        """ Returns the IDs of the participants whose result files were added

            Returns:
                List: list of IDs as Strings
        """

        return list(self.__results.keys())

    def get_test_acc(self):
        """ Returns the maximum accuracy achieved on the training set for each
            participant, whose result file was added

            Returns;
                dict: dictionary that maps the participant's ID to the maximum
                      accuracy achieved on the training set
        """

        return self.get_metric('accuracy')

    def get_val_acc(self):
        """ Returns the maximum accuracy achieved on the validation set for each
            participant, whose result file was added

            Returns;
                dict: dictionary that maps the participant's ID to the maximum
                      accuracy achieved on the validation set
        """

        return self.get_metric('val_accuracy')

    def get_test_loss(self):
        """ Returns the minimum loss achieved on the training set for each
            participant, whose result file was added

            Returns;
                dict: dictionary that maps the participant's ID to the minimum
                      loss achieved on the training set
        """

        return self.get_metric('loss')

    def get_val_loss(self):
        """ Returns the minimum loss achieved on the validation set for each
            participant, whose result file was added

            Returns;
                dict: dictionary that maps the participant's ID to the minimum
                      loss achieved on the validation set
        """

        return self.get_metric('val_loss')


    def get_metric(self, metric):
        """ Returns the metric denoted by the input for each participant, whose
            results file was added

            metric (String):
                The metric that should be retrieved, must be one of 'accuracy',
                'loss', 'val_accuracy' and 'val_loss'

            Returns:
                Dict: dictionary that maps the participant's ID to the metric
        """
        
        assert metric in ['accuracy', 'loss', 'val_accuracy', 'val_loss']

        met = {}
        for p in self.__results.keys():
            met[p] = max(self.__results[p][metric])

        return met
