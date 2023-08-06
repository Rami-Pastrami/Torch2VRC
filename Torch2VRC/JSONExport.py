import json
from pathlib import Path
# A collection of functions related to exporting JSONs that the Unity scripts will use to generate CRTs, materials,
# and other resources


def generate_CRT(location: Path, width: int, height: int, dimensions: str = "2D") -> None:
    """
    Generates a json for Unity to use to generate a CRT of defined dimensions
    :param location: folder location where the json will be saved
    :param width: CRT width
    :param height: CRT height
    :param dimensions: if the CRT should be 2D, cube surface, or 3D
    :return: None
    """
    export_data: dict = {
        "file_type": "CRT",
        "width": width,
        "height": height,
        "dimensions": dimensions
    }
    jsonLocation: Path = location / "CRT.json"

    with open(jsonLocation, "w") as file:
        json.dump(export_data, file)
