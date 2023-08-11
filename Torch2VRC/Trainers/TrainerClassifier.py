from Torch2VRC import Trainers
import torch as pt
import numpy as np
from collections import OrderedDict
from torch import nn
from torch import optim

class TrainerBase(Trainers.TrainerBase):


    def __init__(self, network: pt.nn, raw_training_data: dict, possible_network_outputs: dict):
        super().__init__(network, raw_training_data, possible_network_outputs)


    def generate_counts_of_input_trials(self, trials: dict) -> OrderedDict:
        """
        Returns a dict of counts of the number of trials of each trial name
        :param trials: a (possibly slice) of trials
        :return:
        """
        output: OrderedDict = OrderedDict()
        for trial in trials.keys():
            output[trial] = len(trials[trial])
        return output


    def generate_classifier_testing_input(self, counts_of_trials_keyd_to_answer: OrderedDict) -> pt.Tensor:
        """
        generates a classifiers answer
        :param counts_of_trials_keyd_to_answer:
        :return:
        """
        height: int = 0
        for trial in counts_of_trials_keyd_to_answer.keys():
            height += counts_of_trials_keyd_to_answer[trial]

        arr_out: np.ndarray = np.zeros((len(counts_of_trials_keyd_to_answer), height))

        height = 0
        counter: int = 0
        for trial in counts_of_trials_keyd_to_answer.keys():
            arr_out[counter, height:(height + counts_of_trials_keyd_to_answer[trial])] = 1
            counter += 1
            height += counts_of_trials_keyd_to_answer[trial]

        return pt.Tensor(arr_out)

