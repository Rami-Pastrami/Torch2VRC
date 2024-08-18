from Torch2VRC.Dependencies.Types import CRTDataType, WrapMode
from pathlib import Path
import json


def write_network_JSON(network_details: dict, full_JSON_file__path: Path) -> None:
    with open(full_JSON_file__path, "w") as outfile:
        json.dump(network_details, outfile)

class CRT_definition:
    def __init__(self, X_size: int, Y_size: int, is_double_buffered: bool, data_format: CRTDataType = CRTDataType.RHALF, wrapping: WrapMode = WrapMode.CLAMP):
        self.X_size: int = X_size
        self.Y_size: int = Y_size
        self.is_double_buffered: bool = is_double_buffered
        self.data_format: CRTDataType = data_format
        self.wrapping: WrapMode = wrapping

    def export_as_JSON_dict(self) -> dict:
        output: dict = {
            "x": self.X_size,
            "y": self.Y_size,
            "double_buffered": self.is_double_buffered,
            "color_format": self.data_format.value,
            "wrapping": self.wrapping.value
        }
        return output

class float_array_definition:
    def __init__(self, X_size: int, Y_size: int, flatten: bool):
        self.X_size: int = X_size
        self.Y_size: int = Y_size
        self.flatten: bool = flatten

    def export_as_JSON_dict(self) -> dict:
        output: dict = {
            "x": self.X_size,
            "y": self.Y_size,
            "flatten": self.flatten,
        }
        return output
