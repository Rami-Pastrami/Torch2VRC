from Torch2VRC.Dependencies.Types import ActivationFunction, InputType
from Torch2VRC.Dependencies.UnityExport import CRT_definition, float_array_definition

class AbstractLayerHelper:
    def __init__(self, name: str, x_size: int, y_size: int):
        self.layer_name: str = name
        self.x_size: int = x_size
        self.y_size: int = y_size

    def export_as_JSON_dict(self) -> dict:
        output: dict = {
            "X": self.x_size,
            "Y": self.y_size
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
    def __init__(self, name: str, x_size: int, y_size: int, connects_outward_via_connection_of_name: str, input_type: InputType):
        super().__init__(name, x_size, y_size)
        self.connects_outward_via_connection_of_name = connects_outward_via_connection_of_name
        self.input_type: InputType = input_type

    def export_as_JSON_dict(self) -> dict:
        output: dict = super().export_as_JSON_dict()
        output["connects_to_connection"] = self.connects_outward_via_connection_of_name
        output["data_type"] = self.input_type.value
        output["data_file_name"] = self.input_type.value + ".json"
        return output

    def export_data_as_JSON_dict(self) -> dict:
        if self.input_type == InputType.CRT:
            return self._export_data_as_CRT()
        if self.input_type == InputType.float_array:
            return self._export_data_as_float_data()

class HiddenLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int, connects_outward_via_connection_of_name: str, incoming_activation_function: ActivationFunction):
        super().__init__(name, x_size, y_size)
        self.connects_outward_via_connection_of_name = connects_outward_via_connection_of_name
        self.incoming_activation_function: ActivationFunction = incoming_activation_function

    def export_as_JSON_dict(self) -> dict:
        output: dict = super().export_as_JSON_dict()
        output["connects_to_connection"] = self.connects_outward_via_connection_of_name
        output["incoming_activation_function"] = self.incoming_activation_function.value
        output["data_type"] = InputType.CRT.value  # Hidden layers will always be CRTs
        output["data_file_name"] = InputType.CRT.value + ".json"
        return output


class OutputLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int, incoming_activation_function: ActivationFunction = ActivationFunction.none):
        super().__init__(name, x_size, y_size)
        self.incoming_activation_function: ActivationFunction = incoming_activation_function

    def export_as_JSON_dict(self) -> dict:
        output: dict = super().export_as_JSON_dict()
        output["incoming_activation_function"] = self.incoming_activation_function.value
        output["data_type"] = InputType.CRT.value  # Output layers will always be CRTs
        output["data_file_name"] = InputType.CRT.value + ".json"
        return output
