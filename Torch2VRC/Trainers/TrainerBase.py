import numpy as np
import torch as pt
from torch import nn
from torch import optim

class TrainerBase:

    network: pt.nn
    possible_network_outputs: dict
    raw_training_data: dict
    training_input_tensors_by_layer: dict
    training_output_tensor: pt.Tensor # Also known as testing data


    def __init__(self, network: pt.nn, raw_training_data: dict, possible_network_outputs: dict):
        self.network = network
        self.raw_training_data = raw_training_data
        self.possible_network_outputs = possible_network_outputs

    def set_training_output_key(self) -> None:
        pass

    def _counts_each_trial(self, raw_training_data: dict) -> dict:
        """
        Counts number of trials each option has
        :param raw_training_data: Raw Training Data from Import
        :return: dict with keys being possible choices, and value being the int number of trial occurrences
        """
        counts: dict = {}
        for key in raw_training_data.keys():
            counts[key] = len(raw_training_data[key])
        return counts

    #def train_network





