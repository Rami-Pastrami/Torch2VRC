import torch as pt
from torch import nn
from enum import Enum
from LayerHelpers import LayerHelper, InputLayerHelper




class Torch2VRCWriter():
    def __init__(self, network: pt.nn, input_layer: LayerHelper, layers: list[InputLayerHelper]):
        self.network: pt.nn = network






def process_network_for_output(network: pt.nn) -> Torch2VRCWriter:
    pass