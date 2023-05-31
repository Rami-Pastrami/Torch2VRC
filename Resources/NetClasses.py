import numpy as np
import torch as pt


class TrainingSet():

    def __init__(self, trainingSets: dict, testingSets: list):
        '''
        Generate TrainingSet
        :param trainingSets: where data is fulldict[key = inputName] -> listOfAllTrials[trialNumber] -> ndarray of trial (2D if RNN)
        :param testingSets: where data is fullList[trialNumber] -> ndarray of output
        '''

        randomKey = list(trainingSets.keys())[0]
        if len(trainingSets[randomKey]) != len(testingSets):
            raise Exception("Number of training sets must equal number of testing sets!")

        training: dict = {}

        for index, (key, value) in enumerate(trainingSets):
            training[key] = pt.tensor(np.asarray(value), dtype=pt.float32)

        self.trainingData = training
        self.testingData = pt.tensor(np.asarray(testingSets), dtype=pt.float32)





