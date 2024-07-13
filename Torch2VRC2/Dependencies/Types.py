from enum import Enum

class ActivationFunction(Enum):
    none = 0
    tanH = 1

class InputType(Enum):
    float_array = 1
    CRT = 2

class CRTDataType(Enum):
    RHALF = "RHalf"

class WrapMode(Enum):
    CLAMP = "Clamp"
    REPEAT = "Repeat"
