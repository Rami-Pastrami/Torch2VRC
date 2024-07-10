from collections import OrderedDict
from LayerHelpers import LayerHelper
from pathlib import Path
import torch as pt
import numpy as np
from torch import nn



class AbstractConnectionDefinition:
    def __init__(self, helper: LayerHelper, network: nn):
        connections_raw: OrderedDict = network._modules # Cursed, but works
        if helper.name not in connections_raw.keys():
            raise Exception("Given connection name does not exist in neural network!")
        self.name: str = helper.name
        self.connects_to_layer_of_name: str = helper.connects_to_layer_of_name

    def export_weights_as_png_texture(self, full_file_path: Path):
        raise NotImplementedError("Implement This!")

    def export_biases_as_png_texture(self, full_file_path: Path):
        raise NotImplementedError("Implement This!")


class LinearConnectionDefinition(AbstractConnectionDefinition):
    def __init__(self, helper: LayerHelper, network: pt.nn):
        super(helper, network)

        linear_connection: nn.Linear = network._modules[helper.name]
        self.number_input_neurons: int = linear_connection.in_features
        self.number_output_neurons: int = linear_connection.out_features
        self.weights: np.ndarray = linear_connection.weight.numpy()
        self.biases: np.ndarray = linear_connection.bias.numpy()
