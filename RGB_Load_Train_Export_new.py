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


# layer names of the neural network (think of them as the names of arrays where data is stored between processing),
# mapped to their type
layer_definitions: dict = {
    "inputLayer": "FloatArray1D",
    "hiddenLayer": "CRT1D",
    "outputLayer": "CRT1D"
}

# Connection names and the activation functions they use (MUST MATCH THOSE IN PYTORCH NETWORK)
connection_activations: dict = {
    "innerConnections": "tanh",
    "outerConnections": "none"
}

# Connection mappings (define layer(s) in, and layer out) for each Connection
connectionMappings: dict = {
    "innerConnections": (["inputLayer"], "hiddenLayer"),
    "outerConnections": (["hiddenLayer"], "outputLayer")
}


# import data
imported_log: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")  # in this case, a classifier network being trained,
# so keys in the raw log are repeated a lot, there is a set number of possible answers that each trial can refer to

# Init possible answers from imported log for the classifier network
possible_outputs: list = ["red", "green", "blue", "magenta", "yellow"]

# mapping of which layers get which keys from the training dict (Easy for networks with only 1 input layer, but pay
# attention here if you have multiple different input layers, make sure order matches too
log_keys_mapping_to_layer: dict = {
    "inputLayer": ["red", "green", "blue", "magenta", "yellow"]
}


# LAYER SIZES
NUM_INPUT: int = 3
NUM_HIDDEN: int = 10
NUM_OUTPUT: int = len(possible_outputs)

# Init untrained network
RGB_Net = RGB_NN(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT)






print("convenient breakpoint")