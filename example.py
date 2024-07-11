from pathlib import Path
import pandas as pd
import numpy as np
from  sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn
import torch as pt

from VRCDataImporter.LogFileImporters import RawLogLine, read_log_file_without_further_formatting, segregate_by_tag_to_dataframe_arrays


# This just gets the example file path
current_path: Path = Path(__file__)
current_folder: Path = current_path.parent
file_name: Path = Path("example_log.txt")
example_log_file_path: Path = current_folder.joinpath(file_name)

# load in the log file into a Pandas DataFrame
log_line_objects: list[RawLogLine] = read_log_file_without_further_formatting(example_log_file_path)
data_frame: pd.DataFrame = segregate_by_tag_to_dataframe_arrays(log_line_objects, "color", ["X", "Y", "Z"] )

# Get training / answer sets from the DataFrame
label_encoder: LabelEncoder = LabelEncoder()
one_hot_encoder: OneHotEncoder = OneHotEncoder(sparse_output=False)
data_frame["color_category"] = label_encoder.fit_transform(data_frame["color"])
XYZ_training_data: np.ndarray = data_frame[["X", "Y", "Z"]].to_numpy()
color_answer_data: np.ndarray = data_frame["color_category"].to_numpy()
color_answer_classifier_responses: np.ndarray = one_hot_encoder.fit_transform(color_answer_data.reshape(-1, 1))
color_answer_lookups: np.ndarray = label_encoder.classes_ # what color the int category represents

## Define torch network
class RGB_NN(nn.Module):
    def __init__(self, number_inputs: int, number_neurons_hidden: int, number_answers: int):
        super().__init__()
        self.inner_connections = nn.Linear(number_inputs, number_neurons_hidden)
        self.outer_connections = nn.Linear(number_neurons_hidden, number_answers)

    def forward(self, input_layer: pt.Tensor):
        hidden_layer: pt.Tensor = pt.tanh(self.inner_connections(input_layer))  # notice how the activation is part of this layer
        return self.outer_connections(hidden_layer) # but not here


print("Program End") # easy breakpoint