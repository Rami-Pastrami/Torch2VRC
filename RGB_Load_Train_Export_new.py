from torch import nn
import torch as pt
from pathlib import Path

from Torch2VRC import Loading
from Torch2VRC.Trainers.TrainerClassifier import TrainerClassifier

ASSET_PATH: Path = Path("C:/Users/Rima/Documents/Git Stuff/VRC_NN_RGBTest/Assets/")


# Actual Network Model
class RGB_NN(nn.Module):
    def __init__(self, numInputs: int, numNeuronsHidden: int, numAnswers: int):
        super().__init__()
        self.innerConnections = nn.Linear(numInputs, numNeuronsHidden)
        self.outerConnections = nn.Linear(numNeuronsHidden, numAnswers)

    def forward(self, tensor_input_by_layer_name: dict):
        hiddenLayer: pt.Tensor = pt.tanh(self.innerConnections(tensor_input_by_layer_name["inputLayer"]))  # notice how the activation is part of this layer
        return self.outerConnections(hiddenLayer)


# Init possible answers from imported log for the classifier network
possible_outputs: list = ["red", "green", "blue", "magenta", "yellow"]

# LAYER SIZES
NUM_INPUT: int = 3
NUM_HIDDEN: int = 10
NUM_OUTPUT: int = len(possible_outputs)

# layer names of the neural network (think of them as the names of arrays where data is stored between processing),
# mapped to their type (as they will be in the shader). INPUT LAYER NAMES 'ESPECIALLY' MUST MATCH DICTIONARY KEYS USED
# IN FORWARD FUNCTION
layer_definitions: dict = {
    "inputLayer":
        {
            "type": "FloatArray1D",
            "uniform_name": "_Udon_Input",
            "number_neurons_per_dimension": [NUM_INPUT]
        },
    "hiddenLayer":
        {
            "type": "CRT1D",
            "is_input": False,
            "number_neurons_per_dimension": [NUM_HIDDEN]
        },
    "outputLayer":
        {
            "type": "CRT1D",
            "is_input": False,
            "number_neurons_per_dimension": [NUM_OUTPUT]
        }
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
connection_mappings: dict = {
    "innerConnections": (["inputLayer"], "hiddenLayer"),
    "outerConnections": (["hiddenLayer"], "outputLayer")
}


# Import data from a VRC log file
imported_log: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")
# In this demo, this is a stripped log file, but full logs can be used

# Mapping of which layers get which keys from the training dict (Easy for networks with only 1 input layer, but pay
# attention here if you have multiple different input layers, make sure order matches too. Order Matters
raw_log_keys_mapped_to_input_layers: dict = {
    "inputLayer": ["red", "green", "blue", "magenta", "yellow"]
}
# In this case, there is only 1 input layer called "inputLayer" so that gets all of the keys from the imported logs.


# Init untrained network
RGB_Net = RGB_NN(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT)


# Train Network
# using the classifier trainer for some classifier specific functions
trainer: TrainerClassifier = TrainerClassifier(RGB_Net, imported_log)
trainer.sort_raw_training_data_into_input_tensors(raw_log_keys_mapped_to_input_layers)
trainer.generate_classifier_testing_tensor(possible_outputs)
trainer.train_network()

## Get easy to understand data object for use in exportor (friday)


## Exporter (Sunday)

print("convenient breakpoint")