from torch import nn
import torch as pt
import numpy as np

from Torch2VRC import Loading
from Torch2VRC import Torch2VRC

# Actual Network Model
class RGB_NN(nn.Module):
    def __init__(self, numNeuronsHidden: int, numInputs: int = 3, numAnswers: int = 5):
        super().__init__()
        self.innerConnections = nn.Linear(numInputs, numNeuronsHidden)
        self.outerConnections = nn.Linear(numNeuronsHidden, numAnswers)

    def forward(self, *inputs):
        hiddenLayer: pt.Tensor = pt.tanh(self.innerConnections(inputs[0]))  # add activation function
        return self.outerConnections(hiddenLayer)


# import data
importedLog: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")

# Init NN Builder
RGB_Builder = Torch2VRC.Torch_VRC_Helper(importedLog, [["red", "green", "blue", "magenta", "yellow"]])

# Init NN Network
RGB_Net = RGB_NN(10)

# Train
RGB_Net = RGB_Builder.Train(RGB_Net)

# Export Data
weights, biases = RGB_Builder.ExportNetworkLayersAsNumpy(RGB_Net)

# verification (only to verify answer is sensible
input1 = np.asarray(RGB_Builder.trainingData[0])[0, :]
ouput1 = np.asarray(RGB_Builder.testingData)[0, :]

print("Network results directly: ")
print(str(RGB_Net(RGB_Builder.trainingData[0])[0, :]))
print(str(RGB_Net(RGB_Builder.trainingData[0])[1, :]))
print(str(RGB_Net(RGB_Builder.trainingData[0])[2, :]))

print("convinient breakpoint")