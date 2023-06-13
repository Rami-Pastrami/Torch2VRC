from Torch2VRC import Loading
from Torch2VRC import NetClasses

# import data
importedLog: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")
# separate data into separate channels by color, and prep them for importing into PyTorch

RGB_Net = NetClasses.NetworkDef(3, 10, 5)


# separate into training and testing sets

# Generate testing sets
answerTable, answerCounts = Loading.GenerateTestingData(importedLog)
testing = RGB_Net.GenerateClassifierTestingTensor(answerCounts)

# Generate Training Sets
trainingSet = RGB_Net.GenerateClassifierTrainingTensors(importedLog, [["red", "green", "blue", "magenta", "yellow"]])

# train model

NetClasses.Train(RGB_Net, trainingSet[0], testing)


print("convinient breakpoint")