from pathlib import Path
from Torch2VRC.Layers.SingleCRT.LayerCRTBase import LayerCRTBase
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.JSONExport import generate_CRT


class LayerCRT1D(LayerCRTBase):

    def __init__(self, layer_name: str, network_root: Path, neuron_count_per_dimension: list[int], is_input: bool):
        super().__init__(layer_name, network_root, neuron_count_per_dimension, is_input)
        self.dimensions = LayerBase.NumberOfDimensions.ONE_D

    def generate_unity_file_resources(self) -> None:
        layer_location: Path = self.generate_unity_layer_folder()
        generate_CRT(layer_location, self.neuron_count_per_dimension[0], 1, self.layer_name)