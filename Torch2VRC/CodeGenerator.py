import chevron as ch
from pathlib import Path
from common import Activation, InputSource, LayerTypes

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

    # Following ActivationStr functions return string needed to run activation function
    def ActivationStr_None() -> str:
        return "\n"  # This is just here for code completeness

    def ActivationStr_TanH(inputName: str = "output") -> str:
        return f"output = Activation_Tanh({{inputName}})"


    # Buffer Array Def is used to define input buffers (or lack thereof)
    def BufferArrayDef(size: int, name: str) -> str:
        return "cbuffer BUFFER{ float1 " + name + "[" + str(size) + "]; // Data from CPU"

    def BufferNoDef() -> str:
        return ""

    # Texture input definition (or lack thereof)
    def TexInputDef(varName: str = "_tex") -> str:
        return "uniform sampler2D " + varName

    def TexInputNoDef() -> str:
        return ""

    # The Property functions are used to create the property block
    def Property_TexInput(texName: str = "_TexWeights", texFriendlyName: str = "Weights") -> str:
        return texName + "(\"" + texFriendlyName + "\", 2D = \"black\" {}"  # Fstrings aare a bit messy here

    def Property_ArrayInput() -> str:
        return "\n"

    def GenerateLayerFromMoustache(moustacheFileLocation: Path, NetworkExportFolder: Path,
                                   layerName: str, substitutions: dict) -> None:
        '''
        Writes (overwrites a layer shader file at a given location
        :param moustacheFileLocation: moustache file to read
        :param NetworkExportFolder: folder to export shader to
        :param layerName: layer name
        :param substitutions: substitutions for moustache template
        :return: None
        '''

        if not NetworkExportFolder.exists(): NetworkExportFolder.mkdir()
        shaderFilePath: Path = NetworkExportFolder / (layerName + ".shader")
        with open (moustacheFileLocation, 'r') as template:
            generatedShader: str = ch.render(template, substitutions)

        if shaderFilePath.exists():
            shaderFilePath.unlink()  # Delete shader if already exists to overwrite

        Path.touch(shaderFilePath)
        newShaderFile = open(shaderFilePath, 'w')
        _ = newShaderFile.write(generatedShader)
        newShaderFile.close()


    def GenerateLinearLayerCode(inputConnectionLength: int, outputConnectionLength: int, activationType: Activation,
                                inputType: InputSource, networkName: str, layerName: str,
                                bufferName: str = "_Udon_Buffer") -> None:
        '''
        Generates a Shader / CRT combo running a Linear Layer of the neural network
        :param inputConnectionLength: int - number of input neurons
        :param outputConnectionLength: int = number of output neurons
        :param activationType: Type of activation function (or lack thereof)
        :param isInputArray: bool - if the input is not another CRT but rather an
        :return: None
        '''

        # The GetInput functions are used for defining the input in the for loop

        # Used whenever the input is another CRT
        def GetLayerInputFromTex(texName: str = "_TexWeights") -> str:
            return f"tex2D({{texName}}, float2(IN.localTexcoord.x, 0.5))"

        def GetLayerInputFromArray(udonArrayName: str, indexName: str = "weightX") -> str:
            return f"{{udonArrayName}}[{{indexName}}]"

        # inputs for template generation
        _NETWORK_NAME: str
        _PROPERTY_INPUT: str
        _UDON_BUFFER: str
        _LOOP_INPUT_SOURCE: str
        _INPUT_TEXTURE_DEFINITION: str
        _LAYER_NAME: str
        _NUM_INPUT_NEURONS: str
        _ACTIVATION: str

        _NETWORK_NAME = networkName

        match inputType:
            case InputSource.CRT:
                _PROPERTY_INPUT = Property_TexInput()
                _UDON_BUFFER = BufferNoDef()
                _INPUT_TEXTURE_DEFINITION = TexInputDef()
                _LOOP_INPUT_SOURCE = GetLayerInputFromTex()
            case InputSource.UniformArray:
                _PROPERTY_INPUT = Property_ArrayInput()
                _UDON_BUFFER = BufferArrayDef(inputConnectionLength, bufferName)
                _INPUT_TEXTURE_DEFINITION: str = TexInputNoDef()
                _LOOP_INPUT_SOURCE = GetLayerInputFromArray(bufferName)
        _LAYER_NAME = layerName

        match activationType:
            case Activation.NONE: _ACTIVATION = ActivationStr_None()
            case Activation.TanH: _ACTIVATION = ActivationStr_TanH()



        # TODO Moustache Templating



    pass

