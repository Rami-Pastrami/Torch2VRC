from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase
from Torch2VRC.Connections.ConnectionBase import ConnectionBase
from Torch2VRC.ImageExport import export_np_array_as_png
from Torch2VRC.JSONExport import generate_CRT_definition, generate_material_connection_definition_with_bias, \
    generate_material_connection_definition_without_bias, generate_material_connection_layer_definitions
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

        def get_layer_names_from_layer_array(layers: list[LayerBase]) -> list[str]:
            output: list[str] = []
            for layer in layers:
                output.append(layer.layer_name)
            return output

        # Generate Folder for this specific connection
        self.generate_unity_connection_folder()

        # Export weights (and bias if applicable) as images and get normalizers used to generate the material defnitions

        weight_normalizer: float = export_np_array_as_png( self.weights.detach().numpy(), self.connection_folder / "WEIGHTS.png")

        if self.is_using_bias:
            bias_normalizer = export_np_array_as_png(self.bias.detach().numpy(), self.connection_folder / "BIAS.png")
            generate_material_connection_definition_with_bias(self.connection_folder, weight_normalizer, bias_normalizer)
        else:
            generate_material_connection_definition_without_bias(self.connection_folder, weight_normalizer)

        # Generate CRTs to store weights / biases
        if self.is_using_bias:
            generate_CRT_definition(self.connection_folder, self.weights.detach().numpy().size[0] + 1,
                                    self.weights.detach().numpy().size[1], "weights")
        else:
            generate_CRT_definition(self.connection_folder, self.weights.detach().numpy().size[0],
                                    self.weights.detach().numpy().size[1], "weights")

        # Export what layers are the input / output of this connection
        generate_material_connection_layer_definitions(self.connection_folder,
                                                       get_layer_names_from_layer_array(self.input_layers),
                                                       self.output_layer.layer_name)




        # Generate Shader TODO



        return
        


