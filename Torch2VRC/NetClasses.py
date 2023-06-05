import numpy as np
import torch as pt
from torch import nn


class NetworkDef(nn.Module):

    numberOfLayers: int = 0  # number of layers in the network
    layerOutSizes: list[int]  # The number of neurons each layer outputs
    layerInSizes: list[int]  # The number of neurons each layer receives
    # (as a sum of given input size and additional input data sizes)
    layerInputSizes: list[int]  # number of external input neurons coming it at each layer

    layerActivationFunctions: list  # activation function of a layer
    layerTypes: list  # whether layer is linear, rnn, etc

    _layers: list

    def __init__(self, layerTypes: list[str], layerOutSizes: list[int], layerInSizeBeforeAdditional: list[int],
                 layerInputSizes: list[int], layerActivationFuncs: list):
        '''

        :param layerTypes: type of layer (ex: Linear) as string
        :param layerOutSizes:
        :param layerInSizeBeforeAdditional:
        :param layerInputSizes:
        :param layerActivationFuncs:
        '''

        super().__init__()

        # define base variables
        self.numberOfLayers = len(layerTypes)
        # TODO length check, all lists must be of same length!
        self.layerOutSizes = layerOutSizes
        self.layerInputSizes = layerInputSizes
        self.layerInSizes = [layerInputSizes[i] + layerInSizeBeforeAdditional[i] for i in range(self.numberOfLayers)]
        self.layerActivationFunctions = layerActivationFuncs

        self.layerTypes: list = []
        for L in layerTypes:
            if L.lower() == "linear":
                self.layerTypes.append(nn.Linear)
            # TODO add RNN and other layer types!
            else:
                raise Exception("Unsupported layer type requested!")

        # Define actual layers (Linear, RNN, etc)
        self._layers = []
        for i in range(self.numberOfLayers):
            self._layers.append(self.layerTypes[i](self.layerInSizes[i], self.layerOutSizes[i]))

    def __initOLD__(self, layerSizes: list[list], layerNames: list[str], layerTypes: list, layerActivations: list ):

        super().__init__()

        if len(layerSizes) != len(layerNames):
            raise Exception("Invalid number of neuron sets given number of layers")
        if len(layerTypes) != len(layerTypes):
            raise Exception("Invalid number of layer names given number of layer types")
        if len(layerActivations) != len(layerTypes):
            raise Exception("Invalid number of layer Activations given number of layer types")


        self.numLayers = len(layerNames)
        self.layerNames = layerNames

        self._layerIndexes = list(range(1,self.numLayers)) # used for Forward

        self.layers: dict = {}


        for i in range(len(layerNames)):
            # define layer by name using the layers type, with its number input and output neurons respectfully
            self.layers[layerNames[i]] = layerTypes[i](layerSizes[i][0], layerSizes[i][1])


    def Forward(self, trainingSet: dict) -> pt.Tensor:
        output: pt.Tensor = self._SingleForwardStep(0, )

    def _SingleForwardStep(self, layerIndex: int, inputData: pt.Tensor) -> pt.Tensor:
        return self.layerActivations[layerIndex](self.layers[self.layerNames[layerIndex]](inputData))


class TrainingSet():
    '''
    Holds Training and Testing data for a specific neural network model
    '''

    trainingData: list = None # listOfAllTrials[trialNumber] -> fulldict[key = inputName] -> Tensor of trial (2D if RNN)
    testingData: pt.Tensor = None # fullList[trialNumber] -> Tensor of output

    def __init__(self, trainingSets: list[dict[np.ndarray]], testingSets: list[np.ndarray]):
        '''
        Generate TrainingSet
        :param trainingSets: where data is listOfAllTrials[trialNumber] -> fulldict[key = inputName] -> ndarray of trial (2D if RNN)
        :param testingSets: where data is fullList[trialNumber] -> ndarray of output
        '''


        for i, element in enumerate(trainingSets):
            for j, (key, value) in enumerate(element):
                trainingSets[i][key] = pt.tensor(np.asarray(value), dtype=pt.float32)

        self.trainingData = trainingSets
        self.testingData = pt.Tensor(np.asarray(testingSets), dtype=pt.float32)

    def GetTrial(self, trial: int) -> (dict, pt.Tensor):
        '''
        Returns training and testing data for a specific trial index
        :param trial: int index of trial
        :return: dict of training data, where key is layerName of input and data is the Tensor, and the resultant Tensor
        '''
        return self.trainingData[trial], self.testingData[trial]