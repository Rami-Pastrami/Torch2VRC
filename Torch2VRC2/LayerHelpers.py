from Types import ActivationFunction, InputType
class LayerHelper():
    def __init__(self, name: str, connects_to_layer_of_name: str, activation_function: ActivationFunction):
        self.name: str = name
        self.connects_to_layer_of_name: str = connects_to_layer_of_name
        self.activation_function: ActivationFunction = activation_function

class InputLayerHelper(LayerHelper):
    def __init__(self, name: str, connects_to_layer_of_name: str, activation_function: ActivationFunction, input_type: InputType):
        super().__init__(name, connects_to_layer_of_name, activation_function)
        self.input_type: InputType = input_type