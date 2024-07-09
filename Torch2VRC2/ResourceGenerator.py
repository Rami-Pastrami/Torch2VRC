import torch as pt
from torch import nn
from enum import Enum


class ActivationFunction(Enum):
    tanH = 1

class InputType(Enum):
    float_array = 1
    CRT = 2

class LayerHelper():
    def __init__(self, name: str, connects_to_layer_of_name: str):
        self.name: str = name
        self.connects_to_layer_of_name: str = connects_to_layer_of_name






class Torch2VRCWriter():
    def __init__(self, network: pt.nn):
        self.network: pt.nn = network




def process_network_for_output(network: pt.nn) -> Torch2VRCWriter:
    pass