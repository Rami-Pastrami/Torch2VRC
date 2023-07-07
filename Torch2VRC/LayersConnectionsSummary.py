from Torch2VRC.ImageExport import calculateNormalizer
from Torch2VRC.ImageExport import ExportNPArrayAsPNG
import numpy as np
from pathlib import Path
import json

###################### Layers #######################

# No data held in these! just for context reference

class Layer_FloatArray():
    '''
    Represents a uniform float array that is written to by UDON.
    WARNING, you CANNOT write to this layer, only read from it!
    '''
    layerName: str = None
    ArrayName: str = None
    size: int = None

    def __init__(self, size: int, layerName: str, ArrayName: str):
        self.size = size
        self.layerName = layerName

        nameSlice = ArrayName[0:6]
        if nameSlice != "_Udon_":
            raise Exception("Uniform Float Array Names Must Start With '_Udon_' Exactly!")
        self.ArrayName = ArrayName

class Layer_1D():
    '''
    Standard 1D layer that is held as a CRT
    '''
    layerName: str = None
    size: int = None

    def __init__(self, size: int, layerName: str):

        self.layerName = layerName
        self.size = size


#################### Connections ####################

# These hold data for biases / weights

class Connection_Linear():

    inputSize: int = None
    outputSize: int = None
    connectionName: str = None
    weight: np.ndarray = None
    bias: np.ndarray = None
    weightNormalizer: float = None
    biasNormalizer: float = None
    combinedWeightBias: np.ndarray = None
    activation: str = None  # can be 'none' or 'tanh'

    def __init__(self, weight: np.ndarray, bias: np.ndarray, connectionName: str, activation: str,
                 inputSize: int, outputSize: int):

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weight = weight
        self.connectionName = connectionName  # turn sideways
        self.bias = bias[:, np.newaxis]  # Turn 1D array sideways by making it a 1 wide 2D array

        if (activation != "none") and (activation != "tanh"):
            raise Exception(f"Unknown activation type {activation}!")

        self.activation = activation
        self.weightNormalizer = calculateNormalizer(self.weight)
        self.biasNormalizer = calculateNormalizer(self.bias)


    def ExportConnectionData(self, connectionFolderPath: Path):
        '''
        Exports Linear Weights and Bias as PNGs into the specified folder
        :param folderPath: Folder export path ending with /
        :return: dict containing normalizations for weights and bias
        '''
        ExportNPArrayAsPNG(self.weight, connectionFolderPath / "WEIGHTS.png", self.weightNormalizer)
        ExportNPArrayAsPNG(self.bias, connectionFolderPath / "BIAS.png", self.biasNormalizer)

    def ExportConnectionJSON(self, folderPath: Path):
        '''
        Exports JSON of connection for Unity C# to use to create material and CRT
        :param folderPath:
        :param weightNormalization:
        :param biasNormalization:
        :return:
        '''
        exportData: dict = {}
        exportData["connectionType"] = "linear"
        exportData["width"] = self.inputSize + 1
        exportData["height"] = self.outputSize
        exportData["weightNormalization"] = self.weightNormalizer
        exportData["biasNormalization"] = self.biasNormalizer
        filePath: str = str(folderPath / "DataForMatAndCRT.json")

        with open(filePath, "w") as file:
            json.dump(exportData, file)


################## Network Summary ##################

class Network_Summary():
    pass


