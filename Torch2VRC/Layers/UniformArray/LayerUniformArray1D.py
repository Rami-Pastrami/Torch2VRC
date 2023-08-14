from Torch2VRC.Layers.UniformArray.LayerUniformArrayBase import LayerUniformArrayBase
from Torch2VRC.Layers.LayerBase import LayerBase
from pathlib import Path

class LayerUniformArray1D(LayerUniformArrayBase):


    def __init__(self, layer_name: str, network_root: Path, neuron_count_per_dimension: list[int],
                 array_name: str = "_Udon_input"):
        super().__init__(layer_name, network_root, neuron_count_per_dimension, array_name)
        self.dimensions = LayerBase.NumberOfDimensions.ONE_D

    # TODO any unique export functions