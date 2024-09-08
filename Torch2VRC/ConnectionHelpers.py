from Torch2VRC.Dependencies.Types import ActivationFunction

class AbstractConnectionHelper:
    def __init__(self, connection_name_from_torch: str, target_layer_name: str,
                 outgoing_activation_function: ActivationFunction):
        self.connection_name_from_torch: str = connection_name_from_torch
        self.target_layer_name: str = target_layer_name
        self.outgoing_activation_function: ActivationFunction = outgoing_activation_function


class LinearConnectionHelper(AbstractConnectionHelper):
    def __init__(self, connection_name_from_torch: str, target_layer_name: str, source_layer_name: str,
                 outgoing_activation_function: ActivationFunction):
        super().__init__(connection_name_from_torch,target_layer_name, outgoing_activation_function)
        self.source_layer_name: str = source_layer_name
