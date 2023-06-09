import numpy as np
import torch as pt
from torch import nn
from torch import optim
import copy

class NetworkDef(nn.Module):

    numberOfLayers: int = 0  # number of layers in the network
    layerOutSizes: list[int]  # The number of neurons each layer outputs
    layerInSizes: list[int]  # The number of neurons each layer receives
    # (as a sum of given input size and additional input data sizes)
    layerInputSizes: list[int]  # number of external input neurons coming it at each layer

    layerActivationFunctions: list  # activation function of a layer
    layerTypes: list  # whether layer is linear, rnn, etc

    trainingSet: list
    testing: pt.Tensor

    _mergingLayers: list[bool]  # true where a layer has merging input neurons

    def __init__(self, layerTypes: list[str], layerOutSizes: list[int], layerInSizeBeforeAdditional: list[int],
                 layerInputSizes: list[int], layerActivationFuncs: list, lossFunc: str="MSELoss"):
        '''
        Defines Neural network properties
        :param layerTypes: type of layer (ex: Linear) as string
        :param layerOutSizes: the number of neurons each layer must output
        :param layerInSizeBeforeAdditional: The neurons each layer inputs MINUS any additional external input neurons
        :param layerInputSizes: The number of neurons inputted externally, usually 0 for layers other than the first
        :param layerActivationFuncs: The activation function used for a layer
        :param lossFunc: The loss function to use. Defaults to MSE
        '''

        super(NetworkDef, self).__init__()
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

        if lossFunc.lower() == "mseloss":
            self.lossFunction = nn.MSELoss()

        # Define actual layers (Linear, RNN, etc)
        self._layers = nn.ModuleList() # such that loss can load this
        for i in range(self.numberOfLayers):
            self._layers.append(self.layerTypes[i](self.layerInSizes[i], self.layerOutSizes[i]))

        # Define layers in which input neurons will be merged with output of previous layer
        self._mergingLayers: list = []
        for i in range(self.numberOfLayers):
            self._mergingLayers.append((layerInputSizes[i] != 0))

    def Train(self, numberEpochs: int = 200, learningRate: float = 0.0001):

        print("Starting Training!")

        optimizer = optim.SGD(self.parameters(), lr=learningRate)

        for epochIndex in range(numberEpochs):
            NNOutput = self.Forward(self.trainingSet)
            loss = self.lossFunction(self.trainingSet, NNOutput)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())



    # TODO possible optimization: increment i by 1 to remove need of conditional in GenerateNextInputLayer
    def Forward(self, orderedInputTensors: list[pt.Tensor]) -> pt.Tensor:
        '''
        Runs forward propagation
        :param orderedInputTensors: input Tensors for this particular data sample
        :return: output Tensor
        '''
        previousLayerData: pt.Tensor = None

        for i in range(self.numberOfLayers):
            previousLayerData = self.layerActivationFunctions[i](self._GenerateNextInputLayer(i, previousLayerData,
                                                                                              orderedInputTensors[i]))
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

    def GenerateClassifierTrainingTensors(self, trainingData: dict, keyMappings: list[list or None]) -> list:
        '''
        Converts input data into training Tensors indexed to their matching input layer
        :param trainingData: generated training data dict
        :param keyMappings: list of lists matching the training data keys to their position within and as input layers.
        Example: [[keya1, keya2], None, [keyb1, keyb2]] maps 4 keys to layers 1 and 3, assuming 2 possible outputs a & b

        :return: list of tensors (and nones) indexes to where they will be input
        '''
        output: list = []

        for mappings in keyMappings:

            if mappings == None:
                # No input for this layer
                output.append(None)
                continue

            arrays: list or np.ndarray = []
            for mapping in mappings:
                arrays.append(np.asarray(trainingData[mapping]))
            arrays = np.vstack(arrays)
            output.append(pt.Tensor(arrays))

        self.trainingSet = output
        return output




    def GenerateClassifierTestingTensor(self, numberElementsPerAnswer: list[int]) -> pt.Tensor:
        '''
        Generates a tensor for classifiers (one correct neuron, rest 0)
        :param numberElementsPerAnswer: ordered list of number of samples per category as seen in the training data
        :return:
        '''
        def GenerateSubIndexArray(width: int, index: int) -> np.ndarray:
            o = np.zeros(width)
            o[index] = 1
            return o

        output: list or np.ndarray = []
        aWidth: int = len(numberElementsPerAnswer)

        totalOutElements: int = 0
        for i, e in enumerate(numberElementsPerAnswer):

            for ie in range(e):
                output.append(GenerateSubIndexArray(aWidth, i))

        output = np.asarray(output)
        self.testing = pt.Tensor(output)
        return self.testing
