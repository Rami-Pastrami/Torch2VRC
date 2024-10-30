from Torch2VRC.Dependencies.Types import ActivationFunction

class AbstractConnectionHelper:
    def __init__(self, connection_name_from_torch: str, target_layer_name: str,
                outgoing_activation_function: ActivationFunction, number_neurons_going_in: int,
                 number_neurons_going_out: int):
        self.connection_name_from_torch: str = connection_name_from_torch
        self.target_layer_name: str = target_layer_name
        self.outgoing_activation_function: ActivationFunction = outgoing_activation_function
        self.number_inputs: int = number_neurons_going_in
        self.number_outputs: int = number_neurons_going_out


class LinearConnectionHelper(AbstractConnectionHelper):
    def __init__(self, connection_name_from_torch: str, target_layer_name: str, source_layer_name: str,
                 outgoing_activation_function: ActivationFunction, number_neurons_going_in: int,
                 number_neurons_going_out: int):
        super().__init__(connection_name_from_torch,target_layer_name, outgoing_activation_function)
        self.source_layer_name: str = source_layer_name
        self.number_inputs: int = number_neurons_going_in
        self.number_outputs: int = number_neurons_going_out
