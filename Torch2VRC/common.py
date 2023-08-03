from enum import Enum

# Supported Activation functions
class Activation(Enum):
    NONE = 0
    TanH = 1

# Supported Layer Types
class ConnectionTypes(Enum):
    Linear = 0

# Source of input for layer
class LayerTypes(Enum):
    CRT: 0
    UniformArray = 1

def str2Activation(input: str) -> Activation:
    a: str = input.lower()
    match a:
        case "none": return Activation.NONE
        case "tanh": return Activation.TanH
        case _: raise Exception("Unknown Activation Type!")
