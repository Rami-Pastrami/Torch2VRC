from torch import nn
import torch as pt
from collections import OrderedDict
from pathlib import Path

from Torch2VRC import Loading
from Torch2VRC.Trainers.TrainerClassifier import TrainerClassifier

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

    def forward(self, tensor_input_by_layer_name):
        hiddenLayer: pt.Tensor = pt.tanh(self.innerConnections(tensor_input_by_layer_name["inputLayer"]))  # add activation function
        return self.outerConnections(hiddenLayer)


# layer names of the neural network (think of them as the names of arrays where data is stored between processing),
# mapped to their type (as they will be in the shader). INPUT LAYER NAMES 'ESPECIALLY' MUST MATCH DICTIONARY KEYS USED
# IN FORWARD FUNCTION
layer_definitions: dict = {
    "inputLayer": "FloatArray1D",
    "hiddenLayer": "CRT1D",
    "outputLayer": "CRT1D"
}

# Connection names and the activation functions they use (MUST MATCH THOSE IN PYTORCH NETWORK)
# This is only being done since when applying activation functions within an existing layer function (IE Linear), you
# cannot access the activation function type from the pytorch object. While you can technically separate the activation
# function into a separate layer, doing so within a shader would be greatly inefficient.
connection_activations: dict = {
    "innerConnections": "tanh",
    "outerConnections": "none"
}

# Connection mappings (define layer(s) in, and layer out) for each Connection
connectionMappings: dict = {
    "innerConnections": (["inputLayer"], "hiddenLayer"),
    "outerConnections": (["hiddenLayer"], "outputLayer")
}


# Import data from a VRC log file
imported_log: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")
# In this demo, this is a stripped log file, but full logs can be used

# Init possible answers from imported log for the classifier network
possible_outputs: list = ["red", "green", "blue", "magenta", "yellow"]

# Mapping of which layers get which keys from the training dict (Easy for networks with only 1 input layer, but pay
# attention here if you have multiple different input layers, make sure order matches too. Order Matters
raw_log_keys_mapped_to_input_layers: dict = {
    "inputLayer": ["red", "green", "blue", "magenta", "yellow"]
}
# In this case, there is only 1 input layer called "inputLayer" so that gets all of the keys from the imported logs.


# LAYER SIZES
NUM_INPUT: int = 3
NUM_HIDDEN: int = 10
NUM_OUTPUT: int = len(possible_outputs)


# Init untrained network
RGB_Net = RGB_NN(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT)


# Train Network
trainer = TrainerClassifier(RGB_Net, imported_log, )
# using the classifier trainer for some classifier specific functions

RGB_Net =




print("convenient breakpoint")