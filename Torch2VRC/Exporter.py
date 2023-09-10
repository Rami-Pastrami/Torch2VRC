import torch as pt
from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase
from Torch2VRC.Activation.ActivationFactory import create_activation
from Torch2VRC.Layers.SingleCRT.LayerCRT1D import LayerCRT1D
from Torch2VRC.Layers.UniformArray.LayerUniformArray1D import LayerUniformArray1D
from Torch2VRC.Connections.ConnectionLinear import ConnectionLinear

class Exporter:
    '''
    Imports trained Torch network, as well as some helper definition variables, and packages them in an easy to
    reference object. Also allows for exporting this Network as a set of shaders in Unity
    '''

    #trained_net
    network_root: Path
    layers: dict
    connections: dict



    def __init__(self, trained_net, layer_definitions: dict, unity_project_asset_path: Path, connection_details: dict,
                 network_name: str):
        self.trained_net = trained_net
        self.network_root = unity_project_asset_path / f"Rami-Pastrami/Torch2VRC/{network_name}/"
        self.layers = self._generate_layers(layer_definitions)
        self.connections = self._generate_connections(connection_details)


    def export_to_unity(self, network_name: str) -> None:
        print("Generating Network Folder...")
        self.network_root.mkdir(exist_ok=True)




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
            has_bias = True
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
