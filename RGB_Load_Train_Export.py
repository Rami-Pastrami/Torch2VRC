from torch import nn
import torch as pt

from Torch2VRC import Loading
from Torch2VRC import NetClasses

# import data
importedLog: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")

# Generate testing sets
answerTable, answerCounts = Loading.GenerateTestingData(importedLog)



RGB_Net = NetClasses.NetworkDef(3, 10, 5)


# separate into training and testing sets

testing = RGB_Net.GenerateClassifierTestingTensor(answerCounts)

# Generate Training Sets
trainingSet = RGB_Net.GenerateClassifierTrainingTensors(importedLog, [["red", "green", "blue", "magenta", "yellow"]])

# train model

NetClasses.Train(RGB_Net, trainingSet[0], testing)






# Actual Network Model

class RGB_NN(nn.Module):

    def __init__(self, numNeuronsHidden: int, numInputs: int = 3, numAnswers: int = 5):
        super().__init__()
        self.innerConnections = nn.Linear(numInputs, numNeuronsHidden)
        self.outerConnections = nn.Linear(numNeuronsHidden, numAnswers)

    def forward(self, input: pt.Tensor):
        hiddenLayer: pt.Tensor = pt.tanh(self.innerConnections(input))  # add activation function
        return self.outerConnections(hiddenLayer)



print("convinient breakpoint")