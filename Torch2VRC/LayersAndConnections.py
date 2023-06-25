from Torch2VRC.ImageExport import ExportNPArrayAsPNG
import numpy as np
from pathlib import Path

###################### Layers #######################

# No data held in these! just for context reference

class Layer_FloatArray():
    length: int = None
    ArrayName: str = None
    postConnectionNames: list = []

    def __init__(self, length: int, ArrayName: str, postConnectionNames: list):
        self.length = length
        self.postConnectionNames = postConnectionNames

        nameSlice = ArrayName[0:6]
        if nameSlice != "_Udon_":
            raise Exception("Uniform Float Array Names Must Start With '_Udon_' Exactly!")
        self.ArrayName = ArrayName

class Layer_1D():

    length: int = None
    layerName: str = None
    priorConnectionNames: list = []
    postConnectionNames: list = []

    def __init__(self, length: int, layerName: str, priorConnectionNames: list, postConnectionNames: list):
        self.length = length
        self.layerName = layerName
        self.priorConnectionNames = priorConnectionNames
        self.postConnectionNames = postConnectionNames


#################### Connections ####################

class Connection_Linear():

    connectionName: str = None
    weight: np.ndarray = None
    bias: np.ndarray = None
    combinedWeightBias: np.ndarray = None
    outputLayerName: str = None
    inputLayerName: str = None
    inputDataName: str = None

    def __init__(self, weight: np.ndarray, bias: np.ndarray, connectionName: str, outputLayerName: str,
                 inputLayerName: str, inputDataName: str = None, shouldTransposeBias: bool = True):

        self.weight = weight
        self.outputLayerName = outputLayerName
        self.inputLayerName = inputLayerName
        self.inputDataName = inputDataName
        self.connectionName = connectionName

        if shouldTransposeBias:
            self.bias = bias.transpose()
        else:
            self.bias = bias

    def ExportConnectionData(self, folderPath: str) -> dict:
        '''
        Exports Linear Weights and Bias as PNGs into the specified folder
        :param folderPath: Folder export path ending with /
        :return: dict containing normalizations for weights and bias
        '''
        output: dict = {}
        output["weights"] = ExportNPArrayAsPNG(self.weight, folderPath + "WEIGHTS.png")
        output["bias"] = ExportNPArrayAsPNG(self.bias, folderPath + "BIAS.png")
        return output

    def ExportFull(self, folderPath: str):
        normalizations = self.ExportConnectionData(folderPath)
        # TODO shader data loader export
        # TODO CRT generation