from torch import nn
import torch as pt
from pathlib import Path

from Torch2VRC import Loading
from Torch2VRC.Trainers.TrainerClassifier import TrainerClassifier
from Torch2VRC.Exporter import Exporter

ASSET_PATH: Path = Path("C:/Users/Rima/Documents/Git Stuff/VRC_NN_RGBTest/Assets/")
NETWORK_NAME: str = "test01"

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
            "number_neurons_per_dimension": [NUM_OUTPUT],
        }
}

# Details for each connection the input layer(s), the output layer, the activation function used, and if there is a
# bias
connection_details: dict = {
    "innerConnections":
        {
            "input_layers": ["inputLayer"],
            "output_layer": "hiddenLayer",
            "activation": "tanh",
            "has_bias": True
        },
    "outerConnections":
        {
            "input_layers": ["hiddenLayer"],
            "output_layer": "outputLayer",
            "activation": "none",
            "has_bias": True
        },
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

## Get easy to understand data object for use in exportor
exporter: Exporter = Exporter(RGB_Net, layer_definitions, ASSET_PATH, connection_details, NETWORK_NAME)

## Exporter (Sunday)

print("convenient breakpoint")