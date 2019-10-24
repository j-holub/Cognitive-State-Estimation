import json
import os
import re

import numpy as np

class ResultsHandler:


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
        match = re.search('p\d\d', filename)
        return match.group()

    def get_persons(self):
        return self.__results.keys()

    def get_test_acc(self):
        return self.get_metric('acc')

    def get_val_acc(self):
        return self.get_metric('val_acc')

    def get_test_loss(self):
        return self.get_metric('loss')

    def get_val_loss(self):
        return self.get_metric('val_loss')


    def get_metric(self, metric):
        assert metric in ['acc', 'loss', 'val_acc', 'val_loss']

        met = {}
        for p in self.__results.keys():
            met[p] = max(self.__results[p][metric])

        return met
