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

    def train_network(self, testing_inputs: list[pt.Tensor], testing_predictions: pt.Tensor,
                      number_epochs: int = 2000, learning_rate: float = 0.0001, loss_function=nn.MSELoss()) -> None:
        """
        Trains self.network of up to 3 input layers of complexity
        :param testing_inputs: list of inputs (as Tensors), where each element index MUST be in the same input index
        order for running network model directly
        :param testing_predictions: the singular expected output tenser expected for the input tensor(s)
        :param number_epochs: num epochs to train. Defaults to 2000
        :param learning_rate: Learning rate. Defaults to 0.0001
        :param loss_function: function to use, defaults to nn.MSELoss()
        :return:
        """
        def _Train1Input(net,  numEpochs: int, optimizer, lossFunction, input1: pt.Tensor):

            for epochI in range(numEpochs):
                lossI = lossFunction(testing_predictions, net(input1))
                optimizer.zero_grad()
                lossI.backward()
                optimizer.step()
                print(lossI.item())
            print("Training Complete!")
            return net

        def _Train2Input(net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor, input2: pt.Tensor):

            for epochI in range(numEpochs):
                lossI = lossFunction(testing_predictions, net(input1, input2))
                optimizer.zero_grad()
                lossI.backward()
                optimizer.step()
                print(lossI.item())
            print("Training Complete!")
            return net

        def _Train3Input(net, numEpochs: int, optimizer, lossFunction, input1: pt.Tensor, input2: pt.Tensor,
                         input3: pt.Tensor):

            for epochI in range(numEpochs):
                lossI = lossFunction(testing_predictions, net(input1, input2, input3))
                optimizer.zero_grad()
                lossI.backward()
                optimizer.step()
                print(lossI.item())
            print("Training Complete!")
            return net

        optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)
        number_input_layers: int = len(testing_inputs)

        match(number_input_layers):
            case 1:
                network = _Train1Input(self.network, number_epochs, optimizer, loss_function, testing_inputs[0])
            case 2:
                network = _Train2Input(self.network, number_epochs, optimizer, loss_function, testing_inputs[0],
                                       testing_inputs[1])
            case 3:
                network = _Train3Input(self.network, number_epochs, optimizer, loss_function, testing_inputs[0],
                                       testing_inputs[1], testing_inputs[2])
            case _:
                raise Exception("Unaccounted for number of inputs given!")

        self.network = self.network  # lol