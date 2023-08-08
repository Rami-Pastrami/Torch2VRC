from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase
from Torch2VRC.Connections.ConnectionBase import ConnectionBase
import torch as pt

class ConnectionLinear(ConnectionBase):


    is_using_bias: bool
    weights: pt.Tensor
    bias: pt.Tensor

    def __init__(self, connection_name: str, input_layers: list[LayerBase], output_layer: LayerBase,
                 activation_function: ActivationBase, weights: pt.Tensor, bias: pt.Tensor = None,
                 is_using_bias: bool = True):

        super().__init__(connection_name, input_layers, output_layer, activation_function)
        self.weights = weights
        self.is_using_bias = is_using_bias
        if is_using_bias:
            self.bias = bias




