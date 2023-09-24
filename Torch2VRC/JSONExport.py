import json
from pathlib import Path
# A collection of functions related to exporting JSONs that the Unity scripts will use to generate CRTs, materials,
# and other resources


def generate_CRT(location: Path, width: int, height: int, name: str, dimensions: str = "2D") -> None:
    """
    Generates a json for Unity to use to generate a CRT of defined dimensions
    :param location: folder location where the json will be saved
    :param width: CRT width
    :param height: CRT height
    :param name: name of the CRT
    :param dimensions: if the CRT should be 2D, cube surface, or 3D
    :return: None
    """
    export_data: dict = {
        "file_type": "CRT",
        "width": width,
        "height": height,
        "name": name,
        "dimensions": dimensions
    }

    JSON_location: Path = location / "CRT.json"
    with open(JSON_location, "w") as file:
        json.dump(export_data, file)

def generate_material_texture_load(location: Path, name: str, weightNormalizer: float, biasNormalizer: float = 1.0,
                                   using_bias: bool = True) -> None:
    """
    Exports JSON of connection for Unity C# to use to create material for texture (weights and biases) loading
    :param location: folder location where the json will be saved
    :param weightNormalizer: Normalizer factor for weights
    :param biasNormalizer: Normalizer factor for baises
    :param name: name of the material
    :param using_bias: if a bias is being used, defaults to True
    :return: None
    """
    export_data: dict = {}
    if using_bias:
        export_data = {
            "file_type": "mat",
            "weightNormalizer": weightNormalizer,
            "biasNormalizer": biasNormalizer,
            "name": name,
            "using_bias": True
        }
    else:
        export_data = {
            "file_type": "mat",
            "weightNormalizer": weightNormalizer,
            "name": name,
            "using_bias": False
        }

    JSON_location: Path = location / "texture_load_material.json"
    with open(JSON_location, "w") as file:
        json.dump(export_data, file)


