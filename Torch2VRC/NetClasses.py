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

    def __init__(self, numberInputNeurons: int, numberHiddenNeurons: int, numberOutputNeurons: int):

        super().__init__()
        self.innerConnections = nn.Linear(numberInputNeurons, numberHiddenNeurons)
        self.outerConnections = nn.Linear(numberHiddenNeurons, numberOutputNeurons)

    def forward(self, input: pt.Tensor):
        hiddenLayer: pt.Tensor = pt.tanh(self.innerConnections(input))
        return self.outerConnections(hiddenLayer)
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

def Train(net, trainingData: pt.Tensor, testingData: pt.Tensor, numberEpochs=8000, learningRate=0.0001):

    optimizer = optim.SGD(net.parameters(), lr=learningRate)
    lossFunction = nn.MSELoss()

    for epochI in range(numberEpochs):
        #out = pt.transpose(net(trainingData), 0, 1) #TODO this is a mess
        #lossI = lossFunction(testingData, out)
        lossI = lossFunction(testingData, net(trainingData))
        optimizer.zero_grad()
        lossI.backward()
        optimizer.step()  # update NN weights
        print(lossI.item())