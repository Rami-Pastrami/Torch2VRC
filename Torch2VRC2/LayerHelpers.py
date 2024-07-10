from Torch2VRC2.Dependencies.Types import ActivationFunction, InputType


class AbstractLayerHelper:
    def __init__(self, name: str):
        self.layer_name: str = name


class InputLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, connects_outward_via_connection_of_name: str, input_type: InputType):
        super().__init__(name)
        self.connects_outward_via_connection_of_name = connects_outward_via_connection_of_name
        self.input_type: InputType = input_type


class HiddenLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, connects_outward_via_connection_of_name: str, incoming_activation_function: ActivationFunction):
        super().__init__(name)
        self.connects_outward_via_connection_of_name = connects_outward_via_connection_of_name
        self.incoming_activation_function: ActivationFunction = incoming_activation_function

class OutputLayerHelper(AbstractLayerHelper):
    def __init__(self, name: str, connects_outward_via_connection_of_name: str, incoming_activation_function: ActivationFunction = ActivationFunction.none):
        super().__init__(name)
        self.connects_outward_via_connection_of_name: str = connects_outward_via_connection_of_name
        self.incoming_activation_function: ActivationFunction = incoming_activation_function