from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase

class ConnectionBase:
    """
    Base Connection Class. Not much here, mostly exists as a common root for easy reference
    """

    connection_name: str
    input_layers: list[LayerBase]
    output_layer: LayerBase
    activation_function: ActivationBase
    connection_folder: Path

    def __init__(self, connection_name: str, input_layers: list[LayerBase], output_layer: LayerBase,
                 activation_function: ActivationBase, network_root: Path):

        self.connection_name = connection_name
        self.input_layers = input_layers
        self.output_layer = output_layer
        self.ActivationBase = activation_function
        self.connection_folder = network_root / f"/Connections/{{self.connection_name}}/"

    def generate_unity_file_resources(self) -> dict:
        """
        Generates any required unity resources for the Connection
        :return:
        """
        self.generate_unity_connection_folder()
        return {}

    def generate_unity_connection_folder(self) -> None:
        """
        Generates the folder that the layer file info will be stored under
        :param network_root: Path of the network root
        :return: Path of the folder
        """
        Path.mkdir(self.connection_folder, exist_ok=True)

    def _input_layers_as_strings(self) -> list[str]:
        output: list[str] = []
        for layer in self.input_layers:
            output.append(layer.layer_name)
        return output