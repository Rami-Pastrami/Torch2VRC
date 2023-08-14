from pathlib import Path
from Torch2VRC.Layers.LayerBase import LayerBase

class LayerUniformArrayBase(LayerBase):

    array_name: str  # name of the uniform array the shader calls, should start with "_Udon_"

    def __init__(self, layer_name: str, network_root: Path, neuron_count_per_dimension: list[int],
                 array_name: str = "_Udon_input"):
        super().__init__(layer_name, network_root, neuron_count_per_dimension)
        if array_name[0:6] != "_Udon_":
            raise Exception("All Uniform Array Names must start with '_Udon_' due to VRC restrictions!")
        self.array_name = array_name
        self.is_input = True


    def generate_unity_file_resources(self) -> None:
        # Uniform Arrays do not need anything written to the file system
        hint_file: Path = self.layer_folder / "Float_Array_Layers_Need_No_Files.txt"
        hint_file.touch(exist_ok=True)
