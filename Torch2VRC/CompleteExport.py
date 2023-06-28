import numpy as np
import chevron as cr
from Torch2VRC.NetworkTrainer import Torch_VRC_Helper
from pathlib import Path
import Torch2VRC.CodeGenerator
import shutil


def ExportNetworkToVRC(pathToAssetsFolder: Path, helper: Torch_VRC_Helper, trainedNetwork, networkName: str, initialLayer):

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
    shutil.copy(sourceResources / "NN_Common.cginc", staticResources / "NN_Common.cginc")
    shutil.copy(sourceResources / "LoadConnectionLayer.shader", staticResources / "LoadConnectionLayer.shader")

    # Create Unity Editor Script to handle importing and generation of CRTs, Materials
    Torch2VRC.CodeGenerator.GenerateEditorNetworkImporter(networkRoot, networkName)

    # Export shaders for loading connections

    # Get Network Layout
    networkLayout: list = helper.ExportNetworkLayersNetworkTree(initialLayer, trainedNetwork)

    # Create connection folders
    for connectionName in helper.connectionConnectionsAndActivations.keys():
        connectionPaths[connectionName] = networkRoot / "Connections" / (connectionName + "/")
        connectionPaths[connectionName].mkdir(parents=True, exist_ok=True)

    # Create weight / biases for connections, as well as JSON needed for unity to build materials and CRTs
    for element in networkLayout:
        # if not a connection type, skip
        if type(element).__name__ != "Connection_Linear":
            continue
        element.ExportConnectionData(connectionPaths[element.connectionName])
        element.ExportConnectionJSON(connectionPaths[element.connectionName])

    # Per connection layer
    # Generate PNGs
    # Generate JSONs detailing creation of Material, Data CRT

    # Generate Network Shader
    # Generate Custom Shader File
    # Generate JSON for broad network, connection names

    pass

