from pathlib import Path
import pandas as pd
import numpy as np
import torch.cuda
from  sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn, optim
import torch as pt

from VRCDataImporter.LogFileImporters import RawLogLine, read_log_file_without_further_formatting, segregate_by_tag_to_dataframe_arrays
from Torch2VRC2.LayerHelpers import AbstractLayerHelper, InputLayerHelper, HiddenLayerHelper, OutputLayerHelper
from Torch2VRC2.ConnectionHelpers import AbstractConnectionHelper, LinearConnectionHelper
from Torch2VRC2.ResourceGenerator import Torch2VRCWriter
from Torch2VRC2.Dependencies.Types import ActivationFunction, InputType



NUMBER_HIDDEN_NEURONS_IN_HIDDEN_LAYER: int = 10
NUMBER_EPOCHS: int = 1000
PATH_EXPORT_NETWORK: Path = Path("C:/VRCExport/")

# This just gets the example file path
current_path: Path = Path(__file__)
current_folder: Path = current_path.parent
file_name: Path = Path("example_log.txt")
example_log_file_path: Path = current_folder.joinpath(file_name)
print(f"Using Log file {str(example_log_file_path)}")

# load in the log file into a Pandas DataFrame
log_line_objects: list[RawLogLine] = read_log_file_without_further_formatting(example_log_file_path)
data_frame: pd.DataFrame = segregate_by_tag_to_dataframe_arrays(log_line_objects, "color", ["X", "Y", "Z"] )
print(f"Completed loading log file into initial DataFrame")

# Extract training / answer sets from the DataFrame
label_encoder: LabelEncoder = LabelEncoder()
one_hot_encoder: OneHotEncoder = OneHotEncoder(sparse_output=False)
data_frame["color_category"] = label_encoder.fit_transform(data_frame["color"])
XYZ_training_data: np.ndarray = data_frame[["X", "Y", "Z"]].to_numpy()
color_answer_data: np.ndarray = data_frame["color_category"].to_numpy()
color_answer_classifier_responses: np.ndarray = one_hot_encoder.fit_transform(color_answer_data.reshape(-1, 1))
color_answer_lookups: np.ndarray = label_encoder.classes_ # what color the int category represents
print(f"Extracted the needed information from the DataFrame")

## Define torch network
class PositionToColorGroupNetwork(nn.Module):
    def __init__(self, number_inputs: int, number_neurons_hidden: int, number_answers: int):
        super().__init__()
        self.inner_connections = nn.Linear(number_inputs, number_neurons_hidden)
        self.outer_connections = nn.Linear(number_neurons_hidden, number_answers)

    def forward(self, input_layer: pt.Tensor):
        hidden_layer: pt.Tensor = pt.tanh(self.inner_connections(input_layer))  # notice how the activation is part of this layer
        return self.outer_connections(hidden_layer) # but not here

# XYZ is always 3 inputs, number of hidden neurons can be whatever, number of outputs is the number of possible answers (categories)
NN_RGB: nn.Module = PositionToColorGroupNetwork(3, NUMBER_HIDDEN_NEURONS_IN_HIDDEN_LAYER, len(color_answer_lookups))
print(f"Neural Network has been Defined")

## Train the network

loss_function: nn.Module = nn.MSELoss()
optimizer = optim.Adam(NN_RGB.parameters(), lr = 0.001)

data_tensor: pt.Tensor
response_tensor: pt.Tensor


training_device: str
if torch.cuda.is_available():
    training_device = "cuda"
elif torch.backends.mps.is_available():
    training_device = "mps"
else:
    training_device = "cpu"

print(f"Start training Neural network on device: {training_device}")
NN_RGB.to(training_device) # move model to best training device available
data_tensor = pt.from_numpy(XYZ_training_data).float().to(training_device)
response_tensor = pt.from_numpy(color_answer_classifier_responses).float().to(training_device)

def train_network(network_to_train: nn.Module, training_input: pt.Tensor, expected_output: pt.Tensor,
                  number_epochs: int, loss_object: nn.Module,
                  network_optimizer) -> nn.Module:

    for epoch in range(number_epochs):
        loss = loss_object(network_to_train(training_input), expected_output)
        network_optimizer.zero_grad()
        loss.backward()
        network_optimizer.step()
        print(loss.item())
    return network_to_train

NN_RGB = train_network(NN_RGB, data_tensor, response_tensor, NUMBER_EPOCHS, loss_function, optimizer)

print(f"Neural Network has been trained")

## Define helper definitions to aid in model export

connection_helpers: list[AbstractConnectionHelper] = [ # These have to match the names of the connections from the torch model
    LinearConnectionHelper("inner_connections", "hidden1"),
    LinearConnectionHelper("outer_connections", "output")
]

input_layer_helper: InputLayerHelper = InputLayerHelper("input", "inner_connections", InputType.float_array)
output_layer_helper: OutputLayerHelper = OutputLayerHelper("output")

layer_helpers: list[AbstractLayerHelper] = [
    HiddenLayerHelper("hidden1", "outer_connections", ActivationFunction.tanH),
]

model_exporter: Torch2VRCWriter = Torch2VRCWriter(NN_RGB, input_layer_helper, output_layer_helper, layer_helpers, connection_helpers)


## Output the network for use in Unity / VRC


model_exporter.write_to_unity_directory(PATH_EXPORT_NETWORK, "RGB_Test")



print(f"Neural Network has been output for use in Unity")

print("Program End") # easy breakpoint