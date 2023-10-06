import json
from pathlib import Path
# A collection of functions related to exporting JSONs that the Unity scripts will use to generate CRTs, materials,
# and other resources


def generate_CRT_definition(CRT_save_folder: Path, width: int, height: int, name: str, dimensions: str = "2D") -> None:
    """
    Generates a json for Unity to use to generate a CRT of defined dimensions
    :param CRT_save_folder: folder location where the json will be saved
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

    _write_JSON("CRT", CRT_save_folder, export_data)

def generate_material_connection_definition_with_bias(connection_folder: Path, weight_normalizer: float, bias_normalizer: float) -> None:
    """
    Exports JSON of connection for Unity C# to use to create material for texture (weights and biases) loading
    :param connection_folder: folder location where the json will be saved
    :param weight_normalizer: Normalizer factor for weights
    :param bias_normalizer: Normalizer factor for baises
    :return: None
    """

    export_data: dict = {
        "file_type": "mat",
        "weightNormalizer": weight_normalizer,
        "biasNormalizer": bias_normalizer,
        "using_bias": True
    }

    _write_JSON("load_connection_values", connection_folder, export_data)

def generate_material_connection_definition_without_bias(connection_folder: Path, weight_normalizer: float) -> None:
    """
    Exports JSON of connection for Unity C# to use to create material for texture (weights) loading
    :param connection_folder: folder location where the json will be saved
    :param weight_normalizer: Normalizer factor for weights
    :return: None
    """

    export_data: dict = {
        "file_type": "mat",
        "weightNormalizer": weight_normalizer,
        "using_bias": False
    }

    _write_JSON("load_connection_values", connection_folder, export_data)

def generate_material_connection_layer_definitions(connection_folder: Path, input_layers: list[str], output_layer: str,
                                                  layer_type: str) -> None:
    """
    :param layer_type:
    :param connection_folder:
    :param input_layers:
    :param output_layer:
    :return:
    """
    export_data: dict = {
        "type": layer_type,
        "inputs": input_layers,
        "output": output_layer
    }
    _write_JSON("load_connection_connections", connection_folder, export_data)

def _write_JSON(file_name: str, folder: Path, data: dict) -> None:
    JSON_location: Path = folder / (file_name + ".json")
    with open(JSON_location, "w") as file:
        json.dump(data, file)
