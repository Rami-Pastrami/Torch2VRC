from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase
from Torch2VRC.Connections.ConnectionBase import ConnectionBase
from Torch2VRC.ImageExport import export_np_array_as_png
import torch as pt

class ConnectionLinear(ConnectionBase):
    """
    Linear Connection Mapping, IE a Linear Neural Network 'Layer'
    """

    is_using_bias: bool
    weights: pt.Tensor
    bias: pt.Tensor

    def __init__(self, connection_name: str, input_layers: list[LayerBase], output_layer: LayerBase,
                 activation_function: ActivationBase, network_root: Path, weights: pt.Tensor, bias: pt.Tensor,
                 is_using_bias: bool = True):

        super().__init__(connection_name, input_layers, output_layer, activation_function, network_root)
        self.weights = weights
        self.is_using_bias = is_using_bias
        if is_using_bias:
            self.bias = bias

    def generate_unity_file_resources(self) -> dict:
        """
        Generates any required unity resources for the Connection, including writing weights and biases
        :return:
        """
        super().generate_unity_file_resources()
        output: dict = {"type": "linear"}
        weight_normalizer: float = export_np_array_as_png( self.weights.numpy(), self.connection_folder / "WEIGHTS.png")
        output["weights"] = weight_normalizer

        output["sources"] = self._input_layers_as_strings()
        output["destination"] = self.output_layer.layer_name

        if not self.is_using_bias:
            return output
        bias_normalizer = export_np_array_as_png(self.bias.numpy(), self.connection_folder / "BIAS.png")
        output["bias"] = bias_normalizer
        return output
        


