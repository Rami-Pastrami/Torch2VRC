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
        # establish constant / dependency files that are shared between networks (and by version)
        # export jsons / pngs of all resources
        # write network_definition.json

        pass

