import torch as pt
from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Layers.SingleCRT.LayerCRT1D import LayerCRT1D
from Torch2VRC.Layers.UniformArray.LayerUniformArray1D import LayerUniformArray1D

class Exporter:
    '''
    Imports trained Torch network, as well as some helper definition variables, and packages them in an easy to
    reference object. Also allows for exporting this Network as a set of shaders in Unity
    '''

    #trained_net
    network_root: Path
    layers: dict
    connections: dict



    def __init__(self, trained_net, layer_definitions: dict, unity_project_asset_path: Path, connection_activations: dict, connection_mappings: dict,
                 network_name: str):
        self.trained_net = trained_net
        self.network_root = unity_project_asset_path / f"Rami-Pastrami/Torch2VRC/{network_name}/"
        self.layers = self._generate_layers(layer_definitions)


    def export_to_unity(self, network_name: str) -> None:
        print("Generating Network Folder...")
        self.network_root.mkdir(exist_ok=True)




    def _generate_connections(self, connection_activations: dict, connection_mappings: dict) -> dict:
        output: dict = {}
        net_layers: list = list(self.trained_net.children())

        for connection_name in

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
