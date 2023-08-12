import torch as pt
import numpy as np
from torch import nn
from torch import optim

class TrainerBase:

    network: pt.nn
    raw_training_data: dict
    training_input_tensors_by_input_layer: dict  # ordered in order of pytorch network model training input
    training_output_tensor: pt.Tensor  # Also known as testing data


    def __init__(self, network: pt.nn, raw_training_data: dict):
        self.network = network
        self.raw_training_data = raw_training_data

    def sort_raw_training_data_into_input_tensors(self, raw_log_keys_mapped_to_input_layers: dict) -> None:
        """
        Exports a dict of tensors (keys by input layer name) that each input layer will use for training
        :param raw_log_keys_mapped_to_input_layers:
        :return:
        """

        def _selection_of_training_data_by_key_to_tensor(original_training_data: dict,
                                                         keys_to_export: list[str]) -> pt.Tensor:
            """
            Returns a tensor to be used in a single input layer, given the selection of input log data to use
            :param original_training_data: full imported log data to select from
            :param keys_to_export: list of keys to select from and use for this input layer.
            Ensure widths are consistent
            :return:
            """

            # get needed size of tensor
            tensor_width: int = len(original_training_data[keys_to_export[0]][0])
            tensor_height: int = 0
            for key in keys_to_export:
                tensor_height += len(original_training_data[key])
                if len(original_training_data[key][0]) != tensor_width:
                    raise Exception(f"Width for input key {{key}} is {{len(original_training_data[key][0])}} when expected {{tensor_width}}!")

            # create 2D array of required size
            arr: np.ndarray = np.ndarray([tensor_width, tensor_height])

            # fill in data
            h_index: int = 0
            for key in keys_to_export:
                for trial in original_training_data[key]:
                    arr[:,h_index] = np.asarray(trial)

            # export as tensor
            return pt.Tensor(arr)
        # Generate output dict of tensors
        output: dict = {}
        for input_layer_name in raw_log_keys_mapped_to_input_layers.keys():
            output[input_layer_name] = _selection_of_training_data_by_key_to_tensor(self.raw_training_data,
                                                                                    raw_log_keys_mapped_to_input_layers[input_layer_name])

        # Verify heights of all tensors are consistent, because otherwise training will fail
        height: int = output[0].size(dim=1)
        for input_layer_name in output.keys():
            if output[input_layer_name].size(dim=1) != height:
                raise Exception(f"Input layer {{input_layer_name}} does not have the same height as the other layers, and will fail training as a result")

        self.training_input_tensors_by_input_layer = output

    def train_network(self, number_epochs: int = 2000, learning_rate: float = 0.0001,
                      loss_function=nn.MSELoss()) -> None:
        """
        Trains contained network of up to 3 input layers of complexity
        :param testing_predictions: the singular expected output tenser expected for the input tensor(s)
        :param number_epochs: num epochs to train. Defaults to 2000
        :param learning_rate: Learning rate. Defaults to 0.0001
        :param loss_function: function to use, defaults to nn.MSELoss()
        :return:
        """
        def _training(net,  numEpochs: int, optimizer, lossFunction, inputs: dict, testing_predictions: pt.Tensor):

            for epochI in range(numEpochs):
                lossI = lossFunction(testing_predictions, net(inputs))
                optimizer.zero_grad()
                lossI.backward()
                optimizer.step()
                print(lossI.item())
            print("Training Complete!")
            return net

        optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)

        network = _training(self.network, number_epochs, optimizer, loss_function,
                            self.training_input_tensors_by_input_layer, self.training_output_tensor)
        self.network = network

    def _counts_each_trial(self) -> dict:
        """
        Counts number of trials each option has (reading raw training data)
        :return: dict with keys being possible choices, and value being the int number of trial occurrences
        """
        counts: dict = {}
        for key in self.raw_training_data.keys():
            counts[key] = len(self.raw_training_data[key])
        return counts
