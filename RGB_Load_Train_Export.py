from Torch2VRC import Loading
from Torch2VRC import NetClasses

# import data
importedLog: dict = Loading.LoadLogFileRaw("RGB_Demo_Logs.log")
# separate data into separate channels by color, and prep them for importing into PyTorch

RGB_Net = NetClasses.NetworkDef(["Linear"], [6], [0], [3], ["TanH"])


# separate into training and testing sets





