import torch as pt
import pathlib

class Exporter():

    def __init__(self, trained_net, layer_definitions: dict, connection_activations: dict, connection_mappings: dict):
