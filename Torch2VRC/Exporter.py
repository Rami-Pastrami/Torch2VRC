import torch as pt
from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase
from Torch2VRC.Activation.ActivationFactory import create_activation
from Torch2VRC.Layers.SingleCRT.LayerCRT1D import LayerCRT1D
from Torch2VRC.Layers.UniformArray.LayerUniformArray1D import LayerUniformArray1D
from Torch2VRC.Connections.ConnectionLinear import ConnectionLinear
from Torch2VRC.ShaderExporter import ShaderExport
from Torch2VRC.EditorExporter import EditorExport

class Exporter:
    '''
    Imports trained Torch network, as well as some helper definition variables, and packages them in an easy to
    reference object. Also allows for exporting this Network as a set of shaders in Unity
    '''

    #trained_net

    network_root: Path
    layers: dict
    connections: dict

    STATIC_FOLDER_NAME: str = "Static"

    def __init__(self, trained_net, layer_definitions: dict, unity_project_asset_path: Path, connection_details: dict,
                 network_name: str):
        if network_name.lower() == self.STATIC_FOLDER_NAME.lower():
            raise Exception("Unable to export a network with the same name as the static folder!")

        self.trained_net = trained_net
        self.network_root = unity_project_asset_path / f"Rami-Pastrami/Torch2VRC/{network_name}/"
        self.layers = self._generate_layers(layer_definitions)
        self.connections = self._generate_connections(connection_details)


    def export_to_unity(self) -> None:
        static_folder: Path = self.network_root.parent / self.STATIC_FOLDER_NAME
        static_folder.mkdir(exist_ok=True)
        layer_folder: Path = self.network_root / "Layers"
        layer_folder.mkdir(exist_ok=True)
        connection_folder: Path = self.network_root / "Connections"
        connection_folder.mkdir(exist_ok=True)
        resource_folder: Path = Path.cwd() / "Torch2VRC/Resources"

        # Export Static files (if not already)
        ShaderExport.common_cginc(resource_folder, static_folder)
        ShaderExport.load_linear_weights(resource_folder, static_folder)
        ShaderExport.load_linear_weights_and_biases(resource_folder, static_folder)
        EditorExport.asset_handling(resource_folder, static_folder)

        self._write_layers()
        self._write_connections()


    def _generate_connections(self, connection_details: dict) -> dict:

        output: dict = {}
        net_layers: dict = self.trained_net._modules ## extracts the connection data in a cursed manner

        for connection_name in connection_details.keys():
            connection_data = connection_details[connection_name]
            input_layers: list[LayerBase] = []
            for input_name in connection_data["input_layers"]:
                input_layers.append(self.layers[input_name])
            output_layer: LayerBase = self.layers[connection_data["output_layer"]]
            activation: ActivationBase = create_activation(connection_data["activation"])
            has_bias = True  #TODO
            if "has_bias" in connection_data.keys():
                has_bias = connection_data["has_bias"]
            weights: pt.Tensor = net_layers[connection_name].weight
            bias: pt.Tensor = pt.Tensor()
            if has_bias:
                bias = net_layers[connection_name].bias

            match(type(net_layers[connection_name])):
                case pt.nn.modules.linear.Linear:
                    output[connection_name] = ConnectionLinear(connection_name, input_layers, output_layer, activation,
                                                               self.network_root, weights, bias, is_using_bias=has_bias)
                case _:
                    raise Exception("Unsupported Layer Type!")
        return output

    def _generate_layers(self, layer_definitions: dict) -> dict:
        output: dict = {}
        for layer_name in layer_definitions.keys():
            type: str = layer_definitions[layer_name]["type"]
            layer: LayerBase
            match type:
                case "FloatArray1D":
                    num_neurons_per_dimension: list[int] = layer_definitions[layer_name]["number_neurons_per_dimension"]
                    layer = LayerUniformArray1D(layer_name, self.network_root, num_neurons_per_dimension)
                case "CRT1D":
                    num_neurons_per_dimension: list[int] = layer_definitions[layer_name]["number_neurons_per_dimension"]
                    is_input: bool = layer_definitions[layer_name]["is_input"]
                    layer = LayerCRT1D(layer_name, self.network_root, num_neurons_per_dimension, is_input)
                case _:
                    raise Exception("Unknown Layer Type in input hint!")
            output[layer_name] = layer
        return output

    def _write_layers(self) -> None:
        """
        Writes JSON files that the Unity Plugin uses to generate CRT files for each layer
        :return:
        """
        for layer in self.layers.values():
            layer.generate_unity_file_resources()

    def _write_connections(self) -> None:
        """
        Writes Connection data (weights and biases) and a JSON that explains all connections
        :return:
        """
        connection_context: dict = {}
        for connection in self.connections.values():
            connection_context[connection.connection_name] = connection.generate_unity_file_resources()

