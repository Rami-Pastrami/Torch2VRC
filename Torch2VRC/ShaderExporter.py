from pathlib import Path

class ShaderExport():


    def common_cginc(resources_folder: Path, write_path: Path):
        """
        Copys the NN_Commong.cginc file to the unity project
        :return nothing:
        """
        cginc_file: Path = resources_folder / "NN_Common.cginc"
        destination: Path = write_path / "NN_Common.cginc"
        if not destination.exists():
            destination.write_bytes(cginc_file.read_bytes())


    def load_linear_weights_and_biases(write_path: Path):
        pass


ShaderExport.common_cginc = staticmethod(ShaderExport.common_cginc)

