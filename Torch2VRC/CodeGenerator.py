import chevron as ch
from pathlib import Path

def GenerateEditorNetworkImporter(unityNetworkFolderPath: Path, networkName: str):

    mustachePath: Path = Path.cwd() / "Torch2VRC/Resources/Editor_ImportNetwork.cs.moustache"
    filePath: Path = unityNetworkFolderPath / "Editor_ImportNetwork.cs"
    if filePath.exists(): return  # this file is not dynamic, do don't override if already exists
    with open(mustachePath, 'r') as template:
        generatedText = ch.render(template, {"NetworkName": networkName})

        if filePath.exists():
            filePath.unlink()

        Path.touch(filePath)
        newFile = open(filePath, 'w')
        _ = newFile.write(generatedText)
        newFile.close()

def GenerateNetworkShaders(unityNetworkFolderPath: Path, networkName: str):

    moustachePath: Path = Path.cwd() / "Torch2VRC/Resources/LinearLayer.moustache"

    def GenerateNoActivation() -> str:
        return ""  # This is just here for code completeness

    def GenerateActivationStr_Tanh(inputName: str = "output") -> str:
        return f"output = Activation_Tanh({{inputName}})"

    "sampler2D _inputTex;"

    def GenerateLinearLayerCode(inputConnectionLength: int, outputConnectionLength: int, activationStr: str,
                                isInputArray: bool = False):
        '''
        Generates a Shader / CRT combo running a Linear Layer of the neural network
        :param inputConnectionLength: int - number of input neurons
        :param outputConnectionLength: int = number of output neurons
        :param activationStr: Type of activation function (or lack thereof)
        :param isInputArray: bool - if the input is not another CRT but rather an
        :return:
        '''

        # Used mainly in the start of a layer sequence, to get input data for the layer from a uniform float array
        def GetInputFromArray(udonArrayName: str, indexName: str = "weightX") -> str:
            return f"{{udonArrayName}}[{{indexName}}]"

        # Used whenever the input is another CRT
        def GetInputFromTex(texName: str = "_texInput") -> str:
            return f"tex2D({{texName}}, float2(IN.localTexcoord.x, 0.5))"

        







        pass




    pass

