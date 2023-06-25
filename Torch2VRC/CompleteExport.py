import numpy as np
import chevron as cr
from Torch2VRC.NetworkTrainer import Torch_VRC_Helper
from pathlib import Path


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

