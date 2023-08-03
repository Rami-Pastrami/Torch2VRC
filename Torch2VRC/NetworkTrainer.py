import numpy as np
import torch as pt
from torch import nn
from torch import optim
from Torch2VRC.common import Activation, str2Activation
import Torch2VRC.LayersConnectionsSummary as lac

class Torch_VRC_Helper():

    answerLookup: list  # string list whose index corresponds to output neuron
    trainingData: list[pt.Tensor or None]  # testing input data used for training, in order of usage at location of inputs
    testingData: pt.Tensor  # testing output data used for training
    numInputLayers: int  # the number of separate input layers, NEEDED for hacky training system
    connectionActivations: dict  # Connection names as keys, and their associated pair is the activation function string
    layerObjects: list  # List of Layer definition objects (from LayersConnectionSummary)
    connectionMappings: dict  # connection names as keys, associated pair is a 2 element tuple, with the first being
    # the string name of the layer in, and second element being the string name of the layer out
    networkSummary: lac.Network_Summary  # contains graph and other details summarizing the layout of the network

    def __init__(self,  importedData: dict, answerLookup: list, connectionActivations: dict, layerObjects: list,
                 connectionMappings: dict):

        self.answerLookup = answerLookup
        _ , answerCounts = self.GenerateTestingData(importedData)
        self.trainingData, self.numInputLayers = self.GenerateClassifierTrainingTensors(importedData, answerLookup)
        self.testingData = self.GenerateClassifierTestingTensor(answerCounts)
        self.layerObjects = layerObjects
        self.connectionMappings = connectionMappings

        self.connectionActivations = {}
        for a in connectionActivations.keys():
            self.connectionActivations[a] = str2Activation(connectionActivations[a])


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

        def _Train1Input(net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor):

            for epochI in range(numEpochs):
                lossI = lossFunction(self.testingData, net(input1))
                optimizer.zero_grad()
                lossI.backward()
                optimizer.step()
                print(lossI.item())
            print("Training Complete!")
            return net

        def _Train2Input(net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor, input2: pt.Tensor):

            for epochI in range(numEpochs):
                lossI = lossFunction(self.testingData, net(input1, input2))
                optimizer.zero_grad()
                lossI.backward()
                optimizer.step()
                print(lossI.item())
            print("Training Complete!")
            return net

        def _Train3Input(net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor, input2: pt.Tensor,
                         input3: pt.Tensor):

            for epochI in range(numEpochs):
                lossI = lossFunction(self.testingData, net(input1, input2, input3))
                optimizer.zero_grad()
                lossI.backward()
                optimizer.step()
                print(lossI.item())
            print("Training Complete!")
            return net

        def _returnListWithoutNones(input) -> list:

            output: list = []
            for e in input:
                if e is None: continue
                output.append(e)
            return output


        optimizer = optim.SGD(neuralNetwork.parameters(), lr=learningRate)
        lossFunction = nn.MSELoss()
        inputs: list = _returnListWithoutNones(self.trainingData)

        network = None
        # This is why C like languages are better
        if self.numInputLayers == 1:
            network = _Train1Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0])
        if self.numInputLayers == 2:
            network =  _Train2Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0], inputs[1])
        if self.numInputLayers == 3:
            network =  _Train3Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0], inputs[1], inputs[2])

        if network == None: raise Exception("Unaccounted for number of inputs given!")

        print("Training Complete! Exporting Network Summary...")
        self.networkSummary = self._ExportNetworkAsSummary(network)
        print("Export complete!")
        return network

    def _ExportNetworkAsSummary(self, network) -> lac.Network_Summary:

        def ExtractConnectionsFromNetwork(net, activationsPerConnection: dict) -> dict:
            '''

            :param net: Trained Pytorch network object
            :param activationsPerConnection: dictionary with keys being the connection layer name, and data being the activation function string
            :return:
            '''
            def GetLayerType(specificLayer) -> str:
                ''' Stupid cursed method for finding the layer connection type '''
                layerDef: str = str(specificLayer)
                return layerDef[0: layerDef.find("(")]

            def GenerateLinearConnection(conName: str, activationFunctionName: str, conData) -> lac.Connection_Linear:
                ''' Extracts information from ConnectionData to build Connection_Linear Object '''
                weights: np.ndarray = conData.weight.detach().numpy()
                bias: np.ndarray = conData.bias.detach().numpy().transpose()  # transpose so it fits better in CRT
                inputSize: int = conData.in_features
                outputSize: int = conData.out_features
                return lac.Connection_Linear(weights, bias, conName, activationFunctionName, inputSize, outputSize)

            # TODO add more layer connection types!

            output: dict = {}
            namedModules: list = list(net.named_modules())

            # shitty loop to skip the first index
            for i in range(1, len(namedModules)):
                # define base dict structure
                connectionName: str = namedModules[i][0]
                connectionData = namedModules[i][1]
                activationFunctionName: str = "none"  # Get activation string if defined, else none
                if connectionName in activationsPerConnection.keys():
                    activationFunctionName = activationsPerConnection[connectionName]
                layerType: str = GetLayerType(connectionData)
                if layerType.lower() == "linear":
                    output[connectionName] = GenerateLinearConnection(connectionName, activationFunctionName, connectionData)
            return output

        def LayerArray2LayerDict(layers: list) -> dict:
            '''Converts a list of layers to dictionary of them, key'd by the name of the layer. For my convenience'''

            output: dict = {}
            for layer in layers:
                output[layer.layerName] = layer
            return output

        connections: dict = ExtractConnectionsFromNetwork(network, self.connectionActivations)
        layers: dict = LayerArray2LayerDict(self.layerObjects)
        return lac.Network_Summary(connections, layers, self.connectionMappings)


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