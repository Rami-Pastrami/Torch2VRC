import torch as pt
from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Layers.SingleCRT.LayerCRT1D import LayerCRT1D
from Torch2VRC.Layers.UniformArray.LayerUniformArray1D import LayerUniformArray1D

class Exporter:

    #trained_net
    asset_folder: Path
    network_root_folder: Path
    layers: dict
    connections: dict



    def __init__(self, trained_net, layer_definitions: dict, connection_activations: dict, connection_mappings: dict,
                 unity_project_asset_path: Path, network_name: str):
        self.trained_net = trained_net
        self.asset_folder = unity_project_asset_path
        self.network_root_folder = unity_project_asset_path / f"Rami-Pastrami/Torch2VRC/{network_name}/"
        print("Generating Network Folder...")
        self.network_root_folder.mkdir(exist_ok=True)



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
                    layer = LayerUniformArray1D(layer_name, self.network_root_folder)


