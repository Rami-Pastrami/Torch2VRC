from Torch2VRC import Trainers
import torch as pt
from torch import nn
from torch import optim

class TrainerBase(Trainers.TrainerBase):


    def __init__(self, network: pt.nn, raw_training_data: dict, possible_network_outputs: dict):
        super().__init__(network, raw_training_data, possible_network_outputs)


