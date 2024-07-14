from Torch2VRC2.Dependencies.Types import ActivationFunction, InputType
from Torch2VRC2.Dependencies.UnityExport import CRTDefinition

class AbstractLayerHelper:
    def __init__(self, name: str, x_size: int, y_size: int):
        self.layer_name: str = name
        self.x_size: int = x_size
        self.y_size: int = y_size

    def export_as_JSON_dict(self) -> dict:
        output: dict = {}
        output["name"] = self.layer_name
        output["X"] = self.x_size
        output["Y"] = self.y_size
        return output



class InputLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int, connects_outward_via_connection_of_name: str, input_type: InputType):
        super().__init__(name, x_size, y_size)
        self.connects_outward_via_connection_of_name = connects_outward_via_connection_of_name
        self.input_type: InputType = input_type

    def export_as_JSON_dict(self) -> dict:
        output: dict = super().export_as_JSON_dict()
        output["layer_location"] = "input"
        output["connects_to_connection"] = self.connects_outward_via_connection_of_name
        output["layer_data_type"] = {}
        if self.input_type == InputType.float_array:
            output["float_array"] = {
                "type": "float_array",
                "length": self.x_size * self.y_size
            }
        elif self.input_type == InputType.CRT:
            output["CRT"] = {
                "type": "CRT",
                "CRT_details": CRTDefinition(self.x_size, self.y_size, "Layer_" + self.layer_name)
            }
        return output



class HiddenLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int, connects_outward_via_connection_of_name: str, incoming_activation_function: ActivationFunction):
        super().__init__(name, x_size, y_size)
        self.connects_outward_via_connection_of_name = connects_outward_via_connection_of_name
        self.incoming_activation_function: ActivationFunction = incoming_activation_function

    def export_as_JSON_dict(self) -> dict:
        output: dict = super().export_as_JSON_dict()
        output["layer_location"] = "hidden"
        output["connects_to_connection"] = self.connects_outward_via_connection_of_name
        output["incoming_activation_function"] = self.incoming_activation_function.value
        output["method"] = {
            "CRT_details" : CRTDefinition(self.x_size, self.y_size, "Layer_" + self.layer_name, False).export_as_JSON_dict()
        }
        return output




class OutputLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, x_size: int, y_size: int, incoming_activation_function: ActivationFunction = ActivationFunction.none):
        super().__init__(name, x_size, y_size)
        self.incoming_activation_function: ActivationFunction = incoming_activation_function

    def export_as_JSON_dict(self) -> dict:
        output: dict = super().export_as_JSON_dict()
        output["layer_location"] = "hidden"
        output["incoming_activation_function"] = self.incoming_activation_function.value
        output["method"] = {
            "CRT_details" : CRTDefinition(self.x_size, self.y_size, "Layer_" + self.layer_name, False).export_as_JSON_dict()
        }
        return output