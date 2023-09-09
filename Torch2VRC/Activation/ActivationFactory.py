from Torch2VRC.Activation.ActivationBase import ActivationBase
from Torch2VRC.Activation.ActivationTanH import ActivationTanH
from Torch2VRC.Activation.ActivationNone import ActivationNone


def create_activation(activation_type_name: str, input_array_name: str = "output") -> "ActivationBase":
    match (activation_type_name.lower()):
        case "tanh":
            return ActivationTanH(input_array_name=input_array_name)
        case _:
            # none / invalid
            return ActivationNone()