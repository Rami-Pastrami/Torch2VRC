import numpy as np
import chevron as cr
from Torch2VRC.NetworkTrainer import Torch_VRC_Helper
from pathlib import Path
import Torch2VRC.CodeGenerator
import shutil


def ExportNetworkToVRC(pathToAssetsFolder: Path, helper: Torch_VRC_Helper, networkName: str):

    # Load prereq info into vars
    sourceResources: Path = Path.cwd() / "Torch2VRC/Resources/"

    networkRoot: Path = pathToAssetsFolder / "Rami-Pastrami" / "Torch2VRC" / (networkName + "/")
    staticResources: Path = networkRoot / "Static_Resources/"
    connectionPaths: dict = {}


    # Check if folders exist and confirm action
    print(f"Checking path {str(networkRoot)}...")
    userInput: str = ""
    if networkRoot.is_dir():
        print("Path folder already exists.")
        print("Type ' O' to continue and OVERWRITE any conflicting files.")
        print("Type ' D' to DELETE this directory and start fresh.")
        print("Type anything else to stop the program.")
        userInput = input().lower()
        if (userInput != "o") and (userInput != "d"):
            raise SystemExit(0)
        if userInput == "d":
            networkRoot.unlink(missing_ok=True)
            networkRoot.mkdir(parents=True)

    else:
        print("Path not found. Type 'Y' to create this directory and continue. Anything else will exit")
        userInput = input().lower()
        if userInput != "y":
            raise SystemExit(0)
        networkRoot.mkdir(parents=True)
    del userInput

    # Copy over constant resource files
    print("Copying Static Files...")
    staticResources.mkdir(parents=True, exist_ok=True)
    if not (sourceResources / "NN_Common.cginc").exists:
        shutil.copy(sourceResources / "NN_Common.cginc", staticResources / "NN_Common.cginc")
    if not (sourceResources / "LoadLinearConnectionLayer.shader").exists:
        shutil.copy(sourceResources / "LoadLinearConnectionLayer.shader", staticResources / "LoadLinearConnectionLayer.shader")

    # Create Unity Editor Script to handle importing and generation of CRTs, Materials
    Torch2VRC.CodeGenerator.GenerateEditorNetworkImporter(networkRoot, networkName)



    # Create connection folders
    for connectionName in helper.networkSummary.connections.keys():
        connectionPaths[connectionName] = networkRoot / "Connections" / (connectionName + "/")
        connectionPaths[connectionName].mkdir(parents=True, exist_ok=True)

    # Create weight / biases for connections, as well as JSON needed for unity to build materials and CRTs
    for connection in helper.networkSummary.connections.values():
        connection.ExportConnectionData(connectionPaths[connection.connectionName])
        connection.ExportConnectionJSON(connectionPaths[connection.connectionName])

    # Generate Network Shader


    pass

