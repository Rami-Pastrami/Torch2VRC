from pathlib import Path

class LayerBase:
    """
    Base layer class. Not much here, mostly exists as a common root for easy reference
    """
    layer_name: str
    is_read_only: bool
    is_CRT: bool
    is_float_array: bool
    layer_folder: Path

    def __init__(self, layer_name: str, network_root: Path):
        self.layer_name = layer_name
        self.layer_folder = network_root / f"/Layers/{{self.layer_name}}/"

    def generate_unity_file_resources(self) -> None:
        """
        Generates any required unity resources for the layer (IE, a json to generate a CRT)
        :return: None
        """
        pass

    def generate_unity_layer_folder(self) -> Path:
        """
        Generates the folder that the layer file info will be stored under
        :return: Path of the folder
        """
        Path.mkdir(self.layer_folder, exist_ok=True)
        return self.layer_folder

