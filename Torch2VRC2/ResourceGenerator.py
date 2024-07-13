import torch as pt
from pathlib import Path
from Torch2VRC2.ConnectionHelpers import AbstractConnectionHelper, LinearConnectionHelper
from Torch2VRC2.LayerHelpers import AbstractLayerHelper, InputLayerHelper, OutputLayerHelper
from Torch2VRC2.Dependencies.ConnectionDefinitions import AbstractConnectionDefinition, LinearConnectionDefinition

VERSION: int = 1
'''Used to denote what version we should the denote the export as'''



class Torch2VRCWriter():
    def __init__(self, network: pt.nn, input_layer_helper: InputLayerHelper,
                 output_layer_helper: OutputLayerHelper, hidden_layer_helpers: list[AbstractLayerHelper],
                 connection_helpers: list[AbstractConnectionHelper]):

        self.version: int = VERSION

        self.network: pt.nn = network

        self.first_layer_name: str = input_layer_helper.layer_name
        self.layer_definitions: dict = {}
        self.layer_definitions[input_layer_helper.layer_name] = input_layer_helper
        self.layer_definitions[output_layer_helper.layer_name] = OutputLayerHelper
        for hidden_layer_helper in hidden_layer_helpers:
            self.layer_definitions[hidden_layer_helper.layer_name] = hidden_layer_helper

        self.connection_definitions: dict = {}
        for connection_helper in connection_helpers:
            # Where be my match / switch case?
            if isinstance(connection_helper, LinearConnectionHelper):
                self.connection_definitions[connection_helper.connection_name_from_torch] = LinearConnectionDefinition(self.network, connection_helper)
            else:
                raise Exception("Connection Type not Implemented!")

    def write_to_unity_directory(self, neural_network_folder: Path, name_of_network: str):
        # Establish directories if not already
        if not neural_network_folder.exists():
            neural_network_folder.mkdir()
        connections_dir: Path = self._make_subfolder_if_not_exist(neural_network_folder, "connections")

        # init vars that will be used to build the network.json
        connection_data: dict = {} # store generated normalizers and CRT info during texture import to be saved in the network json
        layer_data: dict = {}

        # establish constant / dependency files that are shared between networks (and by version)


        # export connection weights and biases as textures, get normalizer and CRT information
        connection_data = self._export_connections_and_generate_detail_dict(connections_dir)

        # get CRT

        # write network_definition.json

        pass

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

            # generate normalizers and write the connections as PNG files
            output[connection_name]["normalizers"]["weights"] = self.connection_definitions[connection_name].calculate_weights_png_normalizer()
            self.connection_definitions[connection_name].export_weights_as_png_texture(output[connection_name]["normalizers"]["weights"], connections_dir)
            output[connection_name]["normalizers"]["biases"] = self.connection_definitions[connection_name].calculate_biases_png_normalizer()
            self.connection_definitions[connection_name].export_biases_as_png_texture(output[connection_name]["normalizers"]["weights"], connections_dir)

            # generate CRT details
            output[connection_name]["CRT"] = self.connection_definitions[connection_name].export_CRT_dict_to_hold_connection()


        return output










