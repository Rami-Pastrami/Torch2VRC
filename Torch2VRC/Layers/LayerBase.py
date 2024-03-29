from pathlib import Path
from enum import Enum

class LayerBase:
    """
    Base layer class. Not much here, mostly exists as a common root for easy reference
    """

    class NumberOfDimensions(Enum):
        ONE_D = 1
        TWO_D = 2
        THREE_D = 3

    layer_name: str
    layer_folder: Path
    is_input: bool
    neuron_count_per_dimension: list[int] = []
    dimensions: NumberOfDimensions

    def __init__(self, layer_name: str, network_root: Path, neuron_count_per_dimension: list[int]):
        self.layer_name = layer_name
        layer_dir = network_root / "Layers/"
        self.layer_folder = layer_dir / f"{self.layer_name}/"
        self.neuron_count_per_dimension = neuron_count_per_dimension

    def generate_unity_file_resources(self) -> None:
        """
        Generates any required unity resources for the layer (IE, a json to generate a CRT)
        :return: None
        """
        self.generate_unity_layer_folder()
        pass

    def generate_unity_layer_folder(self) -> None:
        """
        Generates the folder that the layer file info will be stored under
        :return: Path of the folder
        """
        Path.mkdir(self.layer_folder, exist_ok=True)

    @property
    def total_number_neurons(self):
        out: int  = 1
        for d in self.neuron_count_per_dimension:
            out *= d
        return out