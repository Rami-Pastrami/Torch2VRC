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

    _layers: list # actual layers
    _mergingLayers: list[bool]  # true where a layer has merging input neurons

    def __init__(self, layerTypes: list[str], layerOutSizes: list[int], layerInSizeBeforeAdditional: list[int],
                 layerInputSizes: list[int], layerActivationFuncs: list):
        '''
        Defines Neural network properties
        :param layerTypes: type of layer (ex: Linear) as string
        :param layerOutSizes: the number of neurons each layer must output
        :param layerInSizeBeforeAdditional: The neurons each layer inputs MINUS any additional external input neurons
        :param layerInputSizes: The number of neurons inputted externally, usually 0 for layers other than the first
        :param layerActivationFuncs: The activation function used for a layer
        '''

        super().__init__()
        # define base variables
        self.numberOfLayers = len(layerTypes)
        # TODO length check, all lists must be of same length!
        self.layerOutSizes = layerOutSizes
        self.layerInputSizes = layerInputSizes
        self.layerInSizes = [layerInputSizes[i] + layerInSizeBeforeAdditional[i] for i in range(self.numberOfLayers)]

        self.layerTypes: list = []
        for L in layerTypes:
            if L.lower() == "linear":
                self.layerTypes.append(nn.Linear)
            # TODO add RNN and other layer types!
            else:
                raise Exception("Unsupported layer type requested!")

        self.layerActivationFunctions: list = []
        for A in layerActivationFuncs:
            if A.lower() == "tanh":
                self.layerActivationFunctions.append(pt.tanh)
            # TODO add other activation functions!
            else:
                raise Exception("Unsupported Activation Function type requested!")

        # Define actual layers (Linear, RNN, etc)
        self._layers = []
        for i in range(self.numberOfLayers):
            self._layers.append(self.layerTypes[i](self.layerInSizes[i], self.layerOutSizes[i]))

        # Define layers in which input neurons will be merged with output of previous layer
        for i in range(self.numberOfLayers):
            self._mergingLayers.append((layerInputSizes[i] != 0))

    def Forward(self, orderedInputTensors: list[pt.Tensor]) -> pt.Tensor:
        '''
        Runs forward propagation
        :param orderedInputTensors: input Tensors for this particular data sample
        :return: output Tensor
        '''
        previousLayerData: pt.Tensor = None

        for i in range(self.numberOfLayers):
            previousLayerData = self._GenerateNextInputLayer(i, previousLayerData, orderedInputTensors[i])
        return previousLayerData

    def _GenerateNextInputLayer(self, index: int, previousInternalInput: pt.Tensor, previousExternalInput: pt.Tensor) \
            -> pt.Tensor:
        '''
        Generates the next input layer (or the NN output)
        :param index: the layer index
        :param previousInternalInput: the layer from the previous NN layer, may be None
        :param previousExternalInput: input from external input neurons, may be None
        :return: The Tensor to be passed to the next layer (or output entirely)
        '''
        if index == 0:
            # Nothing to merge if this is the first layer!
            return previousExternalInput

        if(self._mergingLayers[index]):
            # we need to merge an input layer
            return pt.cat((previousInternalInput, previousExternalInput), 0)

        # Nothing to merge!
        return previousInternalInput




    def ForwardOld(self, trainingSet: dict) -> pt.Tensor:
        output: pt.Tensor = self._SingleForwardStepOLD(0)

    def _SingleForwardStepOLD(self, layerIndex: int, inputData: pt.Tensor) -> pt.Tensor:
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