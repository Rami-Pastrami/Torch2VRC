from torch import nn
import torch as pt
import numpy as np
from pathlib import Path

from Torch2VRC import Loading
from Torch2VRC import NetworkTrainer
from Torch2VRC import CompleteExport
from Torch2VRC import LayersConnectionsSummary

ASSET_PATH: Path = Path("C:/Users/Rima/Documents/Git Stuff/VRC_NN_RGBTest/Assets/")


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
possibleOutputs = [["red", "green", "blue", "magenta", "yellow"]]


connectionDefs = {
    "innerConnections": {
        "i": "input",
        "o": "hidden",
        "activation": "tanh"
    },
    "outerConnections": {
        "i": "hidden",
        "o": "output",
    }
}
layerDefs = {
    "input": "uniformFloatArray",
    "hidden": "1D",
    "output": "1D"
}

RGB_Builder = NetworkTrainer.Torch_VRC_Helper(importedLog, possibleOutputs, connectionDefs, layerDefs)

# Init NN Network
RGB_Net = RGB_NN(10)

# Train
RGB_Net = RGB_Builder.Train(RGB_Net, numberEpochs=4000)

# # verification (only to verify answer is sensible)
# print("Network results directly: ")
# print(str(RGB_Net(RGB_Builder.trainingData[0])[0, :]))
# print(str(RGB_Net(RGB_Builder.trainingData[0])[1, :]))
# print(str(RGB_Net(RGB_Builder.trainingData[0])[2, :]))
#
# def pythonicNetwork_RGB(weightsIn: dict, biasesIn: dict, input: np.ndarray) -> np.ndarray:
#
#     hiddenLayer: np.ndarray = np.tanh((weightsIn["innerConnections"] @ input) + biasesIn["innerConnections"])
#     output: np.ndarray = (weightsIn["outerConnections"] @ hiddenLayer)  + biasesIn["outerConnections"]
#     return output
#
# print("Pythonic Emulation: ")
# print(str(pythonicNetwork_RGB(weights, biases, np.asarray(RGB_Builder.trainingData[0])[0, :])))
# print(str(pythonicNetwork_RGB(weights, biases, np.asarray(RGB_Builder.trainingData[0])[1, :])))
# print(str(pythonicNetwork_RGB(weights, biases, np.asarray(RGB_Builder.trainingData[0])[2, :])))

#print(normalizers)

# Define initial input Layer
initialLayer = LayersAndConnections.Layer_FloatArray(3, "input", "_Udon_XYZ_In")

# Export
CompleteExport.ExportNetworkToVRC(ASSET_PATH, RGB_Builder, RGB_Net, "Coords2RGB", initialLayer)


print("convenient breakpoint")