from Torch2VRC2.Dependencies.Types import CRTDataType, WrapMode


class CRTDefinition:
    def __init__(self, X_size: int, Y_size: int, name: str, is_double_buffered: bool, format: CRTDataType = CRTDataType.RHALF, wrapping: WrapMode = WrapMode.CLAMP):
        self.X_size: int = X_size
        self.Y_size: int = Y_size
        self.name: str = name
        self.is_double_buffered: bool = is_double_buffered
        self.format_str: str = format.value
        self.wrapping_str: str = wrapping.value

    def export_as_JSON_dict(self) -> dict:
        output: dict = {self.name: {}}
        output[self.name]["x"] = self.X_size
        output[self.name]["y"] = self.Y_size
        output[self.name]["double_buffered"] = self.is_double_buffered
        output[self.name]["color_format"] = self.format_str
        output[self.name]["wrapping"] = self.wrapping_str

