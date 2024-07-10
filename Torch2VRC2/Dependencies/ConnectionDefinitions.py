from collections import OrderedDict
from torch import nn
from pathlib import Path
from LayerHelpers import AbstractLayerHelper
from ConnectionHelpers import AbstractConnectionHelper, LinearConnectionHelper
from Torch2VRC2.Dependencies.ArrayAsPNG import export_np_array_as_png, calculate_normalizer
import torch as pt
import numpy as np




class AbstractConnectionDefinition:
    def __init__(self, connection_helper: AbstractConnectionHelper, network: nn):
        connections_raw: OrderedDict = network._modules # Cursed, but works
        if connection_helper.connection_name_from_torch not in connections_raw.keys():
            raise Exception("Given connection name does not exist in neural network!")
        self.name: str = connection_helper.connection_name_from_torch
        self.connects_to_layer_of_name: str = connection_helper.target_layer_name

    def get_destination_layer(self, generated_layers: dict) -> AbstractLayerHelper:
        if self.connects_to_layer_of_name not in generated_layers:
            raise Exception(f"Unable to find layer object of name {self.connects_to_layer_of_name}!")
        return generated_layers[self.connects_to_layer_of_name]


    def calculate_weights_png_normalizer(self) -> float:
        raise NotImplementedError("Implement This!")


    def export_weights_as_png_texture(self, normalizer: float, full_file_path: Path):
        raise NotImplementedError("Implement This!")


    def calculate_biases_png_normalizer(self) -> float:
        raise NotImplementedError("Implement This!")


    def export_biases_as_png_texture(self, normalizer: float, full_file_path: Path):
        raise NotImplementedError("Implement This!")


class LinearConnectionDefinition(AbstractConnectionDefinition):
    def __init__(self, connection_helper: LinearConnectionHelper, network: pt.nn):
        super(connection_helper, network)

        linear_connection: nn.Linear = network._modules[connection_helper.connection_name_from_torch]
        self.number_input_neurons: int = linear_connection.in_features
        self.number_output_neurons: int = linear_connection.out_features
        self.weights: np.ndarray = linear_connection.weight.numpy()
        self.biases: np.ndarray = linear_connection.bias.numpy()

    def calculate_weights_png_normalizer(self) -> float:
        return calculate_normalizer(self.weights)


    def export_weights_as_png_texture(self, normalizer: float, full_file_path: Path):
        export_np_array_as_png(self.weights, normalizer, full_file_path)


    def calculate_biases_png_normalizer(self) -> float:
        return calculate_normalizer(self.biases)


    def export_biases_as_png_texture(self, normalizer: float, full_file_path: Path):
        export_np_array_as_png(self.biases, normalizer, full_file_path)