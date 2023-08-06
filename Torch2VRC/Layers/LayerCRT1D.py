from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase
from Torch2VRC.JSONExport import generate_CRT

class LayerCRT1D(LayerBase):
    """
    Writable Layer stored as a CRT, 1 Dimensional
    """
    num_neurons: int

    def __init__(self, layer_name: str, num_neurons: int, is_read_only: bool = False):
        super().__init__(layer_name)
        self.num_neurons = num_neurons
        self.is_CRT = True
        self.is_float_array = False
        self.is_read_only = is_read_only


    def generate_unity_file_resources(self, network_root: Path) -> None:
        generate_CRT((network_root / f"Layers/{{self.layer_name}}/"), self.num_neurons, 1)
