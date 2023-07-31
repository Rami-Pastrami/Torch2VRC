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


    def GenerateNoActivation() -> str:
        return ""  # This is just here for code completeness

    def GenerateActivationStr_Tanh(inputName: str = "output") -> str:
        return "output = Activation_Tanh(" + inputName + ")"



    def GenerateLinearLayerCode(inputConnectionLength: int, outputConnectionLength: int, activationStr: str,
                                isInputArray: bool = False):

        # Used mainly in the start of a layer sequence, to get input data for the layer from a uniform float array
        def GetInputFromArray(udonArrayName: str, indexName: str = "weightX") -> str:
            return udonArrayName + "[" + indexName + "]"

        def GetInputFromTex(texName: str = "_texInput") -> str:
            return "tex2D(" + texName + ", float2(IN.localTexcoord.x, 0.5))"



        pass




    pass

