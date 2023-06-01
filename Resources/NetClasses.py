import numpy as np
import torch as pt
from torch import nn


class NetworkDef(nn.Module):



    def __init__(self, layerSizes: list[list], layerNames: list, layerTypes: list):

        super().__init__()

        if len(layerSizes) != len(layerNames):
            raise Exception("Invalid number of neuron sets given number of layers")
        if len(layerTypes) != len(layerTypes):
            raise Exception("Invalid number of layer names given number of layer types")

        self.layers: dict = {}

        for i in range(len(layerNames)):
            # define layer by name using the layers type, with its number input and output neurons respectfully
            self.layers[layerNames[i]] = layerTypes[i](layerSizes[i][0], layerSizes[i][1])


    def Forward(self, input: pt.Tensor) -> pt.Tensor:











class TrainingSet():
    '''
    Holds Training and Testing data for a specific neural network model
    '''



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





