from torch import nn
import torch as pt
import numpy as np
from pathlib import Path

from Torch2VRC import Loading
from Torch2VRC import NetworkTrainer
from Torch2VRC import CompleteExport
from Torch2VRC import LayersConnectionsSummary as lac

ASSET_PATH: Path = Path("C:/Users/Rima/Documents/Git Stuff/VRC_NN_RGBTest/Assets/")


# Actual Network Model
class RGB_NN(nn.Module):
    def __init__(self, numInputs: int, numNeuronsHidden: int, numAnswers: int):
        super().__init__()
        self.innerConnections = nn.Linear(numInputs, numNeuronsHidden)
        self.outerConnections = nn.Linear(numNeuronsHidden, numAnswers)

    def forward(self, *inputs):
        hiddenLayer: pt.Tensor = pt.tanh(self.innerConnections(inputs[0]))  # add activation function
        return self.outerConnections(hiddenLayer)


# Init NN Builder mapping definitions
possibleOutputs = [["red", "green", "blue", "magenta", "yellow"]]

# LAYER SIZES
NUM_INPUT: int = 3
NUM_HIDDEN: int = 10
NUM_OUTPUT: int = len(possibleOutputs)

# Init untrained network
RGB_Net = RGB_NN(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT)

# import data
importedLog: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")

# These will have matching names with the layer names defined in the network
connectionActivations: dict = {
    "innerConnections": "tanh",
    "outerConnections": "none"
}

layerObjects: list = [
    lac.Layer_FloatArray(NUM_INPUT, "inputFromUdon", "_Udon_dataIn"),
    lac.Layer_1D(NUM_HIDDEN, "hiddenLayer"),
    lac.Layer_1D(NUM_OUTPUT, "outputLayer")
]

connectionMappings: dict = {
    "innerConnections": ("inputFromUdon", "hiddenLayer"),
    "outerConnections": ("hiddenLayer", "outputLayer")
}

RGB_Builder = NetworkTrainer.Torch_VRC_Helper(importedLog, possibleOutputs, connectionActivations, layerObjects,
                                              connectionMappings)

# Train
RGB_Net = RGB_Builder.Train(RGB_Net, numberEpochs=4000)


# Export
CompleteExport.ExportNetworkToVRC(ASSET_PATH, RGB_Builder, "Coords2RGB")


print("convenient breakpoint")