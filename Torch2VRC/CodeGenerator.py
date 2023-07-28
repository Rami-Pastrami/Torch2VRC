import chevron as ch
from pathlib import Path

def GenerateEditorNetworkImporter(unityNetworkFolderPath: Path, networkName: str):

    mustachePath: Path = Path.cwd() / "Torch2VRC/Resources/Editor_ImportNetwork.cs.moustache"
    filePath: Path = unityNetworkFolderPath / "Editor_ImportNetwork.cs"
    if filePath.exists(): return
    with open(mustachePath, 'r') as template:
        generatedText = ch.render(template, {"NetworkName": networkName})

        if filePath.exists():
            filePath.unlink()

        Path.touch(filePath)
        newFile = open(filePath, 'w')
        _ = newFile.write(generatedText)
        newFile.close()

def GenerateNetworkShaders(unityNetworkFolderPath: Path, networkName: str):


    def GenerateActivationStr_Tanh() -> str:
        return "output = Activation_Tanh(output)"


    def GenerateLinearLayerCode(inputConnectionLength: int, outputConnectionLength: int, activationStr: str,
                                isInputArray: bool = False):

        pass




    pass

