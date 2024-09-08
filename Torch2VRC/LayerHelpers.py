from Torch2VRC.Dependencies.Types import InputType, LayerType
from Torch2VRC.Dependencies.UnityExport import CRT_definition, float_array_definition


class AbstractLayerHelper:
    def __init__(self, name: str, x_size: int, y_size: int):
        self.layer_name: str = name
        self.x_size: int = x_size
        self.y_size: int = y_size
        self.input_type: InputType = InputType.CRT  # default
        self.layer_type: LayerType = LayerType.HIDDEN # default

    def export_as_JSON_dict(self) -> dict:
        output: dict = {
            "X": self.x_size,
            "Y": self.y_size,
            "data_type": self.input_type.value,
            "layer_type": self.layer_type.value,

        }
        return output

    def export_data_as_JSON_dict(self) -> dict:
        return self._export_data_as_CRT()  # Most common behavior

    def _export_data_as_CRT(self, is_double_buffered: bool = False) -> dict:
        crt: CRT_definition = CRT_definition(self.x_size, self.y_size, is_double_buffered)
        return crt.export_as_JSON_dict()

    def _export_data_as_float_data(self, flatten: bool = False):
        float_array: float_array_definition = float_array_definition(self.x_size, self.y_size, flatten)
        return float_array.export_as_JSON_dict()


class InputLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int, input_type: InputType):
        super().__init__(name, x_size, y_size)
        self.input_type: InputType = input_type
        self.layer_type: LayerType = LayerType.INPUT

    def export_data_as_JSON_dict(self) -> dict:  # OVERRIDDEN
        if self.input_type == InputType.CRT:
            return self._export_data_as_CRT()
        if self.input_type == InputType.float_array:
            return self._export_data_as_float_data()


class HiddenLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int):
        super().__init__(name, x_size, y_size)
        self.layer_type: LayerType = LayerType.HIDDEN
        self.input_type = InputType.CRT  # will always be CRT


class OutputLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int):
        super().__init__(name, x_size, y_size)
        self.layer_type: LayerType = LayerType.OUTPUT
        self.input_type = InputType.CRT # will always be CRT


