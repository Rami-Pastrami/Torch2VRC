import numpy as np
import torch as pt
from torch import nn
from torch import optim
import copy

class Torch_VRC_Helper():

    answerLookup: list  # string list whose index corresponds to output neuron
    trainingData: list[pt.Tensor or None]  # testing input data used for training, in order of usage at location of inputs
    testingData: pt.Tensor  # testing output data used for training
    numInputs: int  # the number of separate input layers

    def __init__(self,  importedData: dict, answerLookup: list):

        self.answerLookup = answerLookup
        _ , answerCounts = self.GenerateTestingData(importedData)
        self.trainingData, self.numInputs = self.GenerateClassifierTrainingTensors(importedData, answerLookup)
        self.testingData = self.GenerateClassifierTestingTensor(answerCounts)

    def GenerateTestingData(self, totalData: dict) -> tuple:
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

    def GenerateClassifierTrainingTensors(self, trainingData: dict, keyMappings: list[list or None]) -> tuple[list, int]:
        '''
        Converts input data into training Tensors indexed to their matching input layer
        :param trainingData: generated training data dict
        :param keyMappings: list of lists matching the training data keys to their position within and as input layers.
        Example: [[keya1, keya2], None, [keyb1, keyb2]] maps 4 keys to layers 1 and 3, assuming 2 possible outputs a & b

        :return: list of tensors (and nones) indexes to where they will be input, number of input layers
        '''
        output: list = []
        numInputLayers: int = 0

        for mappings in keyMappings:
            if mappings == None:
                # No input for this layer
                output.append(None)
                continue

            numInputLayers = numInputLayers + 1

            arrays: list or np.ndarray = []
            for mapping in mappings:
                arrays.append(np.asarray(trainingData[mapping]))
            arrays = np.vstack(arrays)

            output.append(pt.Tensor(arrays))

        self.trainingSet = output
        return (output, numInputLayers)

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
        testing = pt.Tensor(output)
        return testing

    # Not the best way to do this, but due to how pytorch works some things get weird
    def Train(self, neuralNetwork, numberEpochs=2000, learningRate=0.0001):

        '''
        Trains the network, then returns it
        :param neuralNetwork: network you wish to train
        :param numberEpochs: "amount" of training to do. Beware of overfitting!
        :param learningRate: "speed" to learn at
        :return:
        '''

        optimizer = optim.SGD(neuralNetwork.parameters(), lr=learningRate)
        lossFunction = nn.MSELoss()
        inputs: list = self._returnListWithoutNones(self.trainingData)

        # This is why C like languages are better
        if self.numInputs == 1:
            return self._Train1Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0])
        if self.numInputs == 2:
            return self._Train2Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0], inputs[1])
        if self.numInputs == 3:
            return self._Train3Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0], inputs[1], inputs[2])

        raise Exception("Unaccounted for number of inputs given!")

    def ExportNetworkLayersAsNumpy(self, network) -> tuple[dict, dict]:

        '''
        Outputs numpy arrays that make up the inputted neural network
        :param network: input network
        :return: dict of weights (name: data), dict of biases (name: data)
        '''

        weights: dict = {}
        biases: dict = {}
        data: list = list(network.named_modules())

        # shitty loop to skip the first index
        for i in range(1, len(data)):
            # also ensure all arrays are 2D
            wData = data[i][1].weight.detach().numpy()
            if wData.ndim == 1:  # stupid 1D hack
                wData = np.expand_dims(wData, axis=0)
            weights[(data[i][0])] = wData
            bData = data[i][1].bias.detach().numpy()
            if bData.ndim == 1:  # stupid 1D hack
                bData = np.expand_dims(bData, axis=0)
            biases[(data[i][0])] = bData

        return weights, biases

    def _Train1Input(self, net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor):

        for epochI in range(numEpochs):
            lossI = lossFunction(self.testingData, net(input1))
            optimizer.zero_grad()
            lossI.backward()
            optimizer.step()
            print(lossI.item())
        print("Training Complete!")
        return net

    def _Train2Input(self, net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor, input2: pt.Tensor):

        for epochI in range(numEpochs):
            lossI = lossFunction(self.testingData, net(input1, input2))
            optimizer.zero_grad()
            lossI.backward()
            optimizer.step()
            print(lossI.item())
        print("Training Complete!")
        return net

    def _Train3Input(self, net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor, input2: pt.Tensor, input3: pt.Tensor):

        for epochI in range(numEpochs):
            lossI = lossFunction(self.testingData, net(input1, input2, input3))
            optimizer.zero_grad()
            lossI.backward()
            optimizer.step()
            print(lossI.item())
        print("Training Complete!")
        return net

    def _returnListWithoutNones(self, input) -> list:

        output: list = []
        for e in input:
            if e is None: continue
            output.append(e)
        return output


# Example separate Train Function. Otherwise unused
def Train(net, trainingData: pt.Tensor, testingData: pt.Tensor, numberEpochs=2000, learningRate=0.0001):

    optimizer = optim.SGD(net.parameters(), lr=learningRate)
    lossFunction = nn.MSELoss()

    for epochI in range(numberEpochs):
        lossI = lossFunction(testingData, net(trainingData))
        optimizer.zero_grad()
        lossI.backward()
        optimizer.step()  # update NN weights
        print(lossI.item())