from enum import Enum

class ActivationFunction(Enum):
    none = "none"
    tanH = "tanH"

class InputType(Enum):
    float_array = "float_array"
    CRT = "CRT"

class CRTDataType(Enum):
    RHALF = "RHalf"

class WrapMode(Enum):
    CLAMP = "Clamp"
    REPEAT = "Repeat"
