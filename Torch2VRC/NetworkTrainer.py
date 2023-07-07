import numpy as np
import torch as pt
from torch import nn
from torch import optim
import Torch2VRC.LayersConnectionsSummary as lac

class Torch_VRC_Helper():

    answerLookup: list  # string list whose index corresponds to output neuron
    trainingData: list[pt.Tensor or None]  # testing input data used for training, in order of usage at location of inputs
    testingData: pt.Tensor  # testing output data used for training
    numInputLayers: int  # the number of separate input layers
    connectionConnectionsAndActivations: dict  # keys are layer names, data are futher dictionaries where 'i' is an
    # array of input connections, 'o' is an array of output connections, and 'activation' is optional activation type
    layerTypes: dict  # key matched layer name to type (1D, uniformFloatArray, etc)


    def __init__(self,  importedData: dict, answerLookup: list, connectionConnectionsAndActivations: dict,
                 layerTypes: dict):

        self.answerLookup = answerLookup
        _ , answerCounts = self.GenerateTestingData(importedData)
        self.trainingData, self.numInputLayers = self.GenerateClassifierTrainingTensors(importedData, answerLookup)
        self.testingData = self.GenerateClassifierTestingTensor(answerCounts)
        self.connectionConnectionsAndActivations = connectionConnectionsAndActivations
        self.layerTypes = layerTypes

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

        # This is why C like languages are better
        if self.numInputLayers == 1:
            return _Train1Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0])
        if self.numInputLayers == 2:
            return _Train2Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0], inputs[1])
        if self.numInputLayers == 3:
            return _Train3Input(neuralNetwork, numberEpochs, optimizer, lossFunction, inputs[0], inputs[1], inputs[2])

        raise Exception("Unaccounted for number of inputs given!")


    # TODO temporary, switch to a tree structure later!
    def ExportNetworkLayersNetworkTree(self, initialLayer, network) -> list:
        '''
        Outputs structure describing how network is wired together
        :param initialLayer: initial layer object, using class from LayersAndConnections
        :param network: trained PyTorch network
        :return:
        '''

        def _GetLayerType(specificLayer) -> str:
            ''' Stupid cursed method for finding the layer connection type '''
            layerDef: str = str(specificLayer)
            return layerDef[0: layerDef.find("(")]

        namedModules: list = list(network.named_modules())
        output: list = [initialLayer]

        # shitty loop to skip the first index
        for i in range(1, len(namedModules)):
            # define base dict structure
            connectionName: str = namedModules[i][0]
            connectionData = namedModules[i][1]
            layerType = _GetLayerType(connectionData)

            if layerType == "Linear":
                # Get prereqs for connection object
                weights = connectionData.weight.detach().numpy()
                bias = connectionData.bias.detach().numpy().transpose()  # transpose so it fits better in CRT
                inputs: list[str] = self.connectionConnectionsAndActivations[connectionName]["i"]
                outputs: list[str] = self.connectionConnectionsAndActivations[connectionName]["o"]
                inputSize: int = connectionData.in_features
                outputSize: int = connectionData.out_features
                activation: str = "none"
                if "activation" in self.connectionConnectionsAndActivations[connectionName]:
                    activation = self.connectionConnectionsAndActivations[connectionName]["activation"]
                connection = lac.Connection_Linear(weights, bias, connectionName, outputs, inputs, activation,
                                                   inputSize, outputSize)

                # get prereqs for output layer object
                outputLayer = None

                if self.connectionConnectionsAndActivations[connectionName] == "uniformFloatArray":
                    # input is a float array
                    outputLayer = lac.Layer_FloatArray(inputSize, f"{connectionName}_Layer", f"_Udon_{connectionName}")
                elif self.connectionConnectionsAndActivations[connectionName] == "1D":
                    # input is another 1D crt
                    priorConnections: list = [connectionName]  # temp, replace with tree stuff later
                    outputLayer = lac.Layer_1D(inputSize, f"{connectionName}_Layer", priorConnections)

                output.append(connection)
                output.append(outputLayer)
                continue

            raise Exception(f"Unsupported Layer Type {layerType}!")

        return output

    def ExportNetworkAsCustomObject(self, network, connectionMappings: dict, connectionActivations: dict,
                                    layerObjects: list):

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
                ''' Extracts information from ConnectionData to build Conneciton_Linear Object '''
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
                layerType: str = GetLayerType(connectionData)
                activationFunctionName: str = activationsPerConnection[connectionName]
                output[connectionName] = GenerateLinearConnection(connectionName, activationFunctionName, connectionData)
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