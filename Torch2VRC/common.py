from enum import Enum

# Supported Activation functions
class Activation(Enum):
    NONE = 0
    TanH = 1

# Supported Layer Types
class LayerTypes(Enum):
    Linear = 0

# Source of input for layer
class InputSource(Enum):
    CRT: 0
    UniformArray = 1
