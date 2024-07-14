from collections import OrderedDict
from torch import nn
from pathlib import Path
from Torch2VRC2.LayerHelpers import AbstractLayerHelper
from Torch2VRC2.ConnectionHelpers import AbstractConnectionHelper, LinearConnectionHelper
from Torch2VRC2.Dependencies.ArrayAsPNG import export_np_array_as_png, calculate_normalizer
from Torch2VRC2.Dependencies.UnityExport import CRTDefinition
import torch as pt
import numpy as np




class AbstractConnectionDefinition:
    def __init__(self, network: nn, connection_name: str, target_layer_name: str):
        connections_raw: OrderedDict = network._modules # Cursed, but works
        if connection_name not in connections_raw.keys():
            raise Exception("Given connection name does not exist in neural network!")
        self.name: str = connection_name
        self.connects_to_layer_of_name: str = target_layer_name

    def get_destination_layer(self, generated_layers: dict) -> AbstractLayerHelper:
        if self.connects_to_layer_of_name not in generated_layers:
            raise Exception(f"Unable to find layer object of name {self.connects_to_layer_of_name}!")
        return generated_layers[self.connects_to_layer_of_name]


    def calculate_weights_png_normalizer(self) -> float:
        raise NotImplementedError("Implement This!")


    def export_weights_as_png_texture(self, normalizer: float, containing_folder_path: Path):
        raise NotImplementedError("Implement This!")


    def calculate_biases_png_normalizer(self) -> float:
        raise NotImplementedError("Implement This!")


    def export_biases_as_png_texture(self, normalizer: float, containing_folder_path: Path):
        raise NotImplementedError("Implement This!")

    def export_CRT_dict_to_hold_connection(self) -> dict:
        raise NotImplementedError("Implement This!")

    def export_input_mappings(self) -> dict:
        raise NotImplementedError("Implement This!")

class LinearConnectionDefinition(AbstractConnectionDefinition):
    def __init__(self, network: pt.nn, connection_helper: LinearConnectionHelper):
        super().__init__(network, connection_helper.connection_name_from_torch, connection_helper.target_layer_name)

        self.source_layer_name: str = connection_helper.source_layer_name
        linear_connection: nn.Linear = network._modules[connection_helper.connection_name_from_torch]
        self.number_input_neurons: int = linear_connection.in_features
        self.number_output_neurons: int = linear_connection.out_features
        self.weights: np.ndarray = linear_connection.weight.cpu().detach().numpy()
        self.biases: np.ndarray = linear_connection.bias.cpu().detach().numpy().reshape(-1, 1)

    def calculate_weights_png_normalizer(self) -> float:
        return calculate_normalizer(self.weights)


    def export_weights_as_png_texture(self, normalizer: float, containing_folder_path: Path):
        export_np_array_as_png(self.weights, normalizer, containing_folder_path.joinpath(Path(self.name + "_Linear_Weights.png")))


    def calculate_biases_png_normalizer(self) -> float:
        return calculate_normalizer(self.biases)


    def export_biases_as_png_texture(self, normalizer: float, containing_folder_path: Path):
        export_np_array_as_png(self.biases, normalizer, containing_folder_path.joinpath(Path(self.name + "_Linear_Bias.png")))

    def export_CRT_dict_to_hold_connection(self) -> dict:
        return CRTDefinition(self.number_input_neurons + 1, self.number_output_neurons, "Linear_" + self.name, False).export_as_JSON_dict()

    def export_input_mappings(self) -> dict:
        return {"input": self.source_layer_name}