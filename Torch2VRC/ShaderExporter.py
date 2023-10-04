from pathlib import Path

class ShaderExport():


    def common_cginc(resources_folder: Path, write_path: Path):
        """
        Copys the NN_Common.cginc file to the unity project
        :return nothing:
        """
        cginc_file: Path = resources_folder / "NN_Common.cginc"
        destination: Path = write_path / "NN_Common.cginc"
        if not destination.exists():
            destination.write_bytes(cginc_file.read_bytes())

    def load_linear_weights_and_biases(resources_folder: Path, write_path: Path):
        """
        Copys the LoadLinearConnectionLayer.shader file to the unity project
        :return nothing:
        """
        shader_file: Path = resources_folder / "LoadLinearConnectionLayer.shader"
        destination: Path = write_path / "LoadLinearConnectionLayer.shader"
        if not destination.exists():
            destination.write_bytes(shader_file.read_bytes())

    def load_linear_weights(resources_folder: Path, write_path: Path):
        """
        Copys the LoadLinearConnectionLayerNoBias.shader file to the unity project
        :return nothing:
        """
        shader_file: Path = resources_folder / "LoadLinearConnectionLayerNoBias.shader"
        destination: Path = write_path / "LoadLinearConnectionLayerNoBias.shader"
        if not destination.exists():
            destination.write_bytes(shader_file.read_bytes())


ShaderExport.common_cginc = staticmethod(ShaderExport.common_cginc)
ShaderExport.load_linear_weights_and_biases = staticmethod(ShaderExport.load_linear_weights_and_biases)
ShaderExport.load_linear_weights = staticmethod(ShaderExport.load_linear_weights)
