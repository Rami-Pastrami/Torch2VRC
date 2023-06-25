import numpy as np
import chevron as cr
from Torch2VRC.ImageExport import ExportNPArrayAsPNG
from Torch2VRC.NetworkTrainer import Torch_VRC_Helper
from pathlib import Path

class Layer_Linear():

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


def ExportNetworkToVRC(pathToAssetsFolder: Path, helper: Torch_VRC_Helper, trainedNetwork,
                       unityAssetPathToNetwork: str = "Assets/Rami-Pastrami/VRC_NN_RGBTest/Network"):

    # Load prereq info into vars
    weights, biases = helper.ExportNetworkLayersAsNumpy(trainedNetwork)



    # Create Folders if not existant


    # Copy over constant resource files


    # Per connection layer
    # Generate PNGs
    # Generate JSONs detailing creation of Material, Data CRT

    # Generate Network Shader
    # Generate Custom Shader File
    # Generate JSON for broad network, connection names

    pass

