import chevron as ch
from pathlib import Path
from Torch2VRC.common import Activation, ConnectionTypes, LayerTypes

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

def GenerateNetworkShaders(unityNetworkFolderPath: Path, networkName: str, layerObject, layerType: LayerTypes):

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
        shaderFilePath: Path = NetworkExportFolder / ("Layers/" + layerName + "/" + layerName + ".shader")
        with open (moustacheFileLocation, 'r') as template:
            generatedShader: str = ch.render(template, substitutions)

        if shaderFilePath.exists():
            shaderFilePath.unlink()  # Delete shader if already exists to overwrite

        Path.touch(shaderFilePath)
        newShaderFile = open(shaderFilePath, 'w')
        _ = newShaderFile.write(generatedShader)
        newShaderFile.close()


    def GenerateLinearLayerCode(networkUnityFolder: Path, inputConnectionLength: int, outputConnectionLength: int, activationType: Activation,
                                inputType: LayerTypes, networkName: str, layerName: str,
                                bufferName: str = "_Udon_Buffer") -> None:
        '''
        Generates a Shader / CRT combo running a Linear Layer of the neural network
        :param networkUnityFolder: Path - Path to the root network folder within the Unity Project
        :param inputConnectionLength: int - number of input neurons
        :param outputConnectionLength: int = number of output neurons
        :param activationType: Activation - Type of activation function (or lack thereof)
        :param inputType: InputSource - The method of input to this layer
        :param networkName: str - Name of the network
        :param layerName: str = Name of the layer
        :param bufferName: str - if using array input
        :return:
        '''
        # The GetInput functions are used for defining the input in the for loop

        # Used whenever the input is another CRT
        def GetLayerInputFromTex(texName: str = "_TexWeights") -> str:
            return f"tex2D({{texName}}, float2(IN.localTexcoord.x, 0.5))"

        def GetLayerInputFromArray(udonArrayName: str, indexName: str = "weightX") -> str:
            return f"{{udonArrayName}}[{{indexName}}]"

        # inputs for template generation
        _NETWORK_NAME: str = ""
        _PROPERTY_INPUT: str = ""
        _UDON_BUFFER: str = ""
        _LOOP_INPUT_SOURCE: str = ""
        _INPUT_TEXTURE_DEFINITION: str = ""
        _LAYER_NAME: str = ""
        _NUM_INPUT_NEURONS: str = ""
        _NUM_OUTPUT_NEURONS: str = ""
        _ACTIVATION: str = ""


        match inputType:
            case LayerTypes.CRT:
                _PROPERTY_INPUT = Property_TexInput()
                _UDON_BUFFER = BufferNoDef()
                _INPUT_TEXTURE_DEFINITION = TexInputDef()
                _LOOP_INPUT_SOURCE = GetLayerInputFromTex()
            case LayerTypes.UniformArray:
                _PROPERTY_INPUT = Property_ArrayInput()
                _UDON_BUFFER = BufferArrayDef(inputConnectionLength, bufferName)
                _INPUT_TEXTURE_DEFINITION: str = TexInputNoDef()
                _LOOP_INPUT_SOURCE = GetLayerInputFromArray(bufferName)
            case _:
                raise Exception("Unknown Input Type!")

        match activationType:
            case Activation.NONE: _ACTIVATION = ActivationStr_None()
            case Activation.TanH: _ACTIVATION = ActivationStr_TanH()
            case _: raise Exception("Unknown Activation Type!")

        _LAYER_NAME = layerName
        _NETWORK_NAME = networkName
        _NUM_INPUT_NEURONS = str(inputConnectionLength)
        _NUM_OUTPUT_NEURONS = str(outputConnectionLength)

        substitutions: dict = {
            "NETWORK_NAME": _NETWORK_NAME,
            "PROPERTY_INPUT": _PROPERTY_INPUT,
            "UDON_BUFFER": _UDON_BUFFER,
            "LOOP_INPUT_SOURCE": _LOOP_INPUT_SOURCE,
            "INPUT_TEXTURE_DEFINITION": _INPUT_TEXTURE_DEFINITION,
            "LAYER_NAME": _LAYER_NAME,
            "NUM_INPUT_NEURONS": _NUM_INPUT_NEURONS,
            "NUM_OUTPUT_NEURONS": _NUM_OUTPUT_NEURONS,
            "ACTIVATION": _ACTIVATION
        }

        moustachePath: Path = Path.cwd() / "Torch2VRC/Resources/LinearLayer.moustache"

        GenerateLayerFromMoustache(moustachePath, networkUnityFolder, layerName, substitutions)

    # TODO loop through layer objects, skip array "layers"

    # Skip any


    pass