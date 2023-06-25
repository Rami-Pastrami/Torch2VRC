from Torch2VRC.ImageExport import ExportNPArrayAsPNGAndGetNormalizer
import numpy as np
from pathlib import Path

###################### Layers #######################

# No data held in these! just for context reference

class Layer_Base():
    postConnectionNames: list = []
    layerName: str = None

class Layer_FloatArray(Layer_Base):

    ArrayName: str = None
    size: int = None

    def __init__(self, size: int, layerName: str, ArrayName: str):
        self.size = size
        self.layerName = layerName

        nameSlice = ArrayName[0:6]
        if nameSlice != "_Udon_":
            raise Exception("Uniform Float Array Names Must Start With '_Udon_' Exactly!")
        self.ArrayName = ArrayName

class Layer_1D(Layer_Base):

    priorConnectionNames: list = []
    size: int = None

    def __init__(self, size: int, layerName: str, priorConnectionNames: list):

        self.layerName = layerName
        self.priorConnectionNames = priorConnectionNames
        self.size = size


#################### Connections ####################

# These hold data for biases / weights

class Connection_Linear():

    inputSize: int = None
    outputSize: int = None
    connectionName: str = None
    weight: np.ndarray = None
    bias: np.ndarray = None
    combinedWeightBias: np.ndarray = None
    outputNames: list[str] = None
    inputNames: list[str] = None
    activation: str = None  # can be none or tanh

    def __init__(self, weight: np.ndarray, bias: np.ndarray, connectionName: str, outputNames: list[str],
                 inputNames: list[str], activation: str, inputSize: int, outputSize: int):

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weight = weight
        self.outputNames = outputNames
        self.inputNames = inputNames
        self.connectionName = connectionName
        self.bias = bias

        if (activation != "none") or (activation != "tanh"):
            raise Exception(f"Unknown activation type {activation}!")

        self.activation = activation


    def ExportConnectionData(self, connectionFolderPath: Path) -> dict:
        '''
        Exports Linear Weights and Bias as PNGs into the specified folder
        :param folderPath: Folder export path ending with /
        :return: dict containing normalizations for weights and bias
        '''
        output: dict = {}
        output["weights"] = ExportNPArrayAsPNGAndGetNormalizer(self.weight, connectionFolderPath / "WEIGHTS.png")
        output["bias"] = ExportNPArrayAsPNGAndGetNormalizer(self.bias, connectionFolderPath / "BIAS.png")
        return output

    def ExportFull(self, folderPath: Path):
        normalizations = self.ExportConnectionData(folderPath)
        # TODO shader data loader export
        # TODO CRT generation