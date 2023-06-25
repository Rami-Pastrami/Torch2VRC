import numpy as np
import chevron as cr
from Torch2VRC.NetworkTrainer import Torch_VRC_Helper
from pathlib import Path


def ExportNetworkToVRC(pathToAssetsFolder: Path, helper: Torch_VRC_Helper, trainedNetwork, networkName: str):

    # Load prereq info into vars
    NetworkRoot: Path = pathToAssetsFolder / "Rami-Pastrami" / "Torch2VRC" / (networkName + "/")

    # Check if folders exist and confirm action
    print(f"Checking path {str(NetworkRoot)}...")
    userInput: str = ""
    if NetworkRoot.is_dir():
        print("Path folder already exists.")
        print("Type ' O' to continue and OVERWRITE any conflicting files.")
        print("Type ' D' to DELETE this directory and start fresh.")
        print("Type anything else to stop the program.")
        userInput = input().lower()
        if userInput != "o" or userInput != "d":
            raise SystemExit(0)
        if userInput == "d":
            NetworkRoot.unlink(missing_ok=True)
            NetworkRoot.mkdir(parents=True)

    else:
        print("Path not found. Type 'Y' to create this directory and continue. Anything else will exit")
        userInput = input().lower()
        if userInput != "y":
            raise SystemExit(0)
        NetworkRoot.mkdir(parents=True)
    del userInput

    

    # Create Folders if not existant


    # Copy over constant resource files


    # Per connection layer
    # Generate PNGs
    # Generate JSONs detailing creation of Material, Data CRT

    # Generate Network Shader
    # Generate Custom Shader File
    # Generate JSON for broad network, connection names

    pass

