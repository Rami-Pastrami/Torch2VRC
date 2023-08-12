from Torch2VRC import Trainers
import torch as pt
import numpy as np
from collections import OrderedDict
from torch import nn
from torch import optim

class TrainerClassifier(Trainers.TrainerBase):


    def __init__(self, network: pt.nn, raw_training_data: dict):
        super().__init__(network, raw_training_data)

    def generate_classifier_testing_tensor(self, possible_outputs: list[str]) -> pt.Tensor:

        width: int = len(possible_outputs)
        total_height: int = 0

        # calculate total height
        for key in self.raw_training_data.keys():
            total_height += len(self.raw_training_data[key])

        # create 2D array of required size
        arr: np.ndarray = np.zeros([width, total_height])

        # set correct values to 1
        index: int = 0
        current_starting_height: int = 0
        for possible_output in possible_outputs:
            arr[index, current_starting_height: len(self.raw_training_data[possible_output]) + current_starting_height] = 1
            current_starting_height += len(self.raw_training_data[possible_output])

        self.training_output_tensor = pt.Tensor(arr)
        return self.training_output_tensor


