import numpy as np
import torch as pt
from torch import nn
from torch import optim
import copy

class NNDataBuilder():

    answerLookup: list  # string list whose index corresponds to output neuron
    trainingData: list[pt.Tensor]  # testing input data used for training, in order of usage at location of inputs
    testingData: pt.Tensor  # testing output data used for training


    def __init__(self,  importedData: dict):

        self.answerLookup, answerCounts = self.GenerateTestingData(importedData)
        self.trainingData = self.GenerateClassifierTrainingTensors(importedData, self.answerLookup)
        self.testingData = self.GenerateClassifierTestingTensor(answerCounts)

    def GenerateTestingData(totalData: dict) -> tuple:
        '''
        Generates inputs required for generating testing data for neural network right from Log output
        :param totalData: Log file output
        :return: Tuple(ordered list of answers) & (corresponding count of each answer in the set)
        '''

        answers: list = list(totalData.keys())
        counts: list = []
        for ans in answers:
            counts.append(len(totalData[ans]))
        return (answers, counts)

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