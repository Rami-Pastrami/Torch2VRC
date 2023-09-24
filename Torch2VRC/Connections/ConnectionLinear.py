from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase
from Torch2VRC.Connections.ConnectionBase import ConnectionBase
from Torch2VRC.ImageExport import export_np_array_as_png
from Torch2VRC.JSONExport import generate_material_texture_load
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

    def generate_unity_file_resources(self) -> None:
        """
        Generates any required unity resources for the Connection, including writing weights and biases
        :return:
        """
        # Generate Folder
        super().generate_unity_file_resources()

        # Generate Material (JSON) TO load in weights and biases, and the CRTs to store the data in

        output: dict = {"type": "linear"}
        weight_normalizer: float = export_np_array_as_png( self.weights.detach().numpy(), self.connection_folder / "WEIGHTS.png")
        output["weight_normalizer"] = weight_normalizer

        output["sources"] = self._input_layers_as_strings()
        output["destination"] = self.output_layer.layer_name


        if  self.is_using_bias:
            bias_normalizer = export_np_array_as_png(self.bias.detach().numpy(), self.connection_folder / "BIAS.png")
            output["bias"] = bias_normalizer

        # Generate Shader to connect the layers together

        # Generate Material (JSON)

        return
        


