from pathlib import Path

class LayerBase:
    """
    Base layer class. Not much here, mostly exists as a common root for easy reference
    """
    layer_name: str
    is_read_only: bool
    is_CRT: bool
    is_float_array: bool


    def __init__(self, layer_name: str):
        self.layer_name = layer_name

    def generate_unity_file_resources(self, network_root: Path) -> None:
        """
        Generates any required unity resources for the layer (IE, a json to generate a CRT)
        :param network_root: root folder of the network within the Unity project (relative to system, not unity)
        :return:
        """
        pass