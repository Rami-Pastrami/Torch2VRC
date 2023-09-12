from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase

class LayerCRTBase(LayerBase):

    def __init__(self, layer_name: str, network_root: Path, neuron_count_per_dimension: list[int], is_input: bool):
        super().__init__(layer_name, network_root, neuron_count_per_dimension)
        self.is_input = is_input

