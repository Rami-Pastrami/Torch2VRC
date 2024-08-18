import torch as pt
from pathlib import Path
import json
import time
from Torch2VRC.ConnectionHelpers import AbstractConnectionHelper, LinearConnectionHelper
from Torch2VRC.LayerHelpers import AbstractLayerHelper
from Torch2VRC.Dependencies.ConnectionDefinitions import LinearConnectionDefinition
from Torch2VRC.Dependencies.UnityExport import write_network_JSON

VERSION: int = 2
'''Used to denote what version we should the denote the export as'''



class Torch2VRCWriter():
    def __init__(self, network: pt.nn, network_name: str, layer_helpers: list[AbstractLayerHelper],
                 connection_helpers: list[AbstractConnectionHelper]):

        self.version: int = VERSION
        self.network_name = network_name
        self.network: pt.nn = network

        self.layer_definitions: dict = {}
        for layer_helper in layer_helpers:
            self.layer_definitions[layer_helper.layer_name] = layer_helper
        self.connection_definitions: dict = {}
        for connection_helper in connection_helpers:
            # Where be my match / switch case?
            if isinstance(connection_helper, LinearConnectionHelper):
                self.connection_definitions[connection_helper.connection_name_from_torch] = LinearConnectionDefinition(self.network, connection_helper)
            else:
                raise Exception("Connection Type not Implemented!")


    def write_to_unity_directory(self, root_neural_networks_folder: Path, name_of_network: str):
        """
        Creates the needed folders, and exports the properties as JSONs and data as Images
        """

        # Establish directories if not already existing
        if not root_neural_networks_folder.exists():
            root_neural_networks_folder.mkdir()
        network_folder: Path = self._make_subfolder_if_not_exist(root_neural_networks_folder, name_of_network)
        layers_folder: Path = self._make_subfolder_if_not_exist(network_folder, "Layers")
        connections_folder: Path = self._make_subfolder_if_not_exist(network_folder, "Connections")
        linear_connections_folder: Path = self._make_subfolder_if_not_exist(connections_folder, "Linear")
        #TODO Other types of connection folders as they get added

        self._write_network_json(Path.joinpath(network_folder, "network.json"))



        return
        #TODO remove old
        connections_dir: Path = self._make_subfolder_if_not_exist(root_neural_networks_folder, "connections")

        # init vars that will be used to build the network.json
        connection_data: dict = {} # store generated normalizers and CRT info during texture import to be saved in the network json
        layer_data: dict = {}

        # establish constant / dependency files that are shared between networks (and by version)


        # export connection weights and biases as textures, get normalizer and CRT information
        connection_data = self._export_connections_and_generate_detail_dict(connections_dir)

        # export layer details
        layer_data = self._export_layer_details()

        # write network_definition.json
        network_dict = {
            "network_name": self.network_name,
            "layers": layer_data,
            "connections": connection_data
        }







    def _write_network_json(self, network_path: Path) -> None:
        network: dict = {
            "name": self.network_name,
            "exporter_version": VERSION,
            "export_timestamp": int(time.time()),
            "layer_names": [layer_name for layer_name in self.layer_definitions.keys()],
            "connection_linear_names": [connection.name for connection in self.connection_definitions.values() if isinstance(connection, LinearConnectionDefinition)]
            #TODO Add more connection layer types here
        }
        write_JSON(network, network_path)

    def _make_subfolder_if_not_exist(self, parent_directory: Path, subfolder_name: str) -> Path:
        subfolder: Path = parent_directory.joinpath(subfolder_name + "/")
        if not subfolder.exists():
            subfolder.mkdir()
        return subfolder

    def _export_connections_and_generate_detail_dict(self, connections_dir: Path) -> dict:
        output: dict = {}
        for connection_name in self.connection_definitions:
            output[connection_name] = {}
            output[connection_name]["normalizers"] = {}
            output[connection_name]["CRT"] = {}
            output[connection_name]["inputs"] = {}
            output[connection_name]["output"] = ""

            # generate normalizers and write the connections as PNG files
            output[connection_name]["normalizers"]["weights"] = self.connection_definitions[connection_name].calculate_weights_png_normalizer()
            self.connection_definitions[connection_name].export_weights_as_png_texture(output[connection_name]["normalizers"]["weights"], connections_dir)
            output[connection_name]["normalizers"]["biases"] = self.connection_definitions[connection_name].calculate_biases_png_normalizer()
            self.connection_definitions[connection_name].export_biases_as_png_texture(output[connection_name]["normalizers"]["weights"], connections_dir)

            # generate CRT details
            output[connection_name]["CRT"] = self.connection_definitions[connection_name].export_CRT_dict_to_hold_connection()

            # Define the input layer(s)
            output[connection_name]["inputs"] = self.connection_definitions[connection_name].export_input_mappings()

            # Define the output layer
            output[connection_name]["output"] = self.connection_definitions[connection_name].connects_to_layer_of_name


        return output

    def _export_layer_details(self) -> dict:
        output: dict = {}
        for layer_name in self.layer_definitions:
            output[layer_name] = self.layer_definitions[layer_name].export_as_JSON_dict()
        return output


def write_JSON(dictionary: dict, full_JSON_file_path: Path) -> None:
    with open(full_JSON_file_path, "w") as outfile:
        json.dump(dictionary, outfile)