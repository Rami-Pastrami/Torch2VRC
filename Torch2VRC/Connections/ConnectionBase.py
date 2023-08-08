from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.Activation.ActivationBase import ActivationBase

class ConnectionBase:


    connection_name: str
    input_layers: list[LayerBase]
    output_layer: LayerBase
    activation_function: ActivationBase

    def __init__(self, connection_name: str, input_layers: list[LayerBase], output_layer: LayerBase,
                 activation_function: ActivationBase):

        self.connection_name = connection_name
        self.input_layers = input_layers
        self.output_layer = output_layer
        self.ActivationBase = activation_function

    def generate_unity_file_resources(self, network_root: Path) -> None:
        """
        Generates any required unity resources for the Connection (IE, a json to generate a CRT)
        :param network_root: root folder of the network within the Unity project (relative to system, not unity)
        :return:
        """
        pass

    def generate_unity_connection_folder(self, network_root: Path) -> Path:
        """
        Generates the folder that the layer file info will be stored under
        :param network_root: Path of the network root
        :return: Path of the folder
        """
        connection_folder: Path = network_root / f"/Connections/{{self.connection_name}}/"
        Path.mkdir(connection_folder, exist_ok=True)
        return connection_folder

