# Responsible for exporting Pytorch weights as PNGs that can be read in via HLSL shaders
import numpy as np
import math
from PIL import Image as im


# given weight and bias dictionaries, saves each as a PNG and exports dict of normalizations
def ExportLayersBiases(weights: dict, biases: dict, folderPath: str = "") -> dict:

    normalizers: dict = {}
    # Weights
    for w in weights.keys():
        normalizers[w] = ExportNPArrayAsPNG(weights[w], folderPath + w + "_WEIGHTS.png")
    for b in biases.keys():
        normalizers[b] = ExportNPArrayAsPNG(biases[b], folderPath + b + "_BIASES.png")
    return normalizers

def ExportNPArrayAsPNG(inputArray: np.ndarray, filePathName: str) -> float:

    '''
    Saves Layer Numpy Array as a PNG image, and returns the normalizer needed to rescale it to original values
    :param inputArray: 2D numpy array from PyTorch itself
    :param filePathName: file name / path to save PNG at
    :return: normalizer value
    '''

    normalizer: float = _calculateNormalizer(inputArray)
    RGBAArray: np.ndarray = _NumpyLayerToRGBAArray( inputArray, normalizer)
    ImageData = im.fromarray(RGBAArray, mode="RGBA")
    ImageData.save(filePathName)

    return normalizer

def _NumpyLayerToRGBAArray( layer: np.ndarray, normalizer: float) -> np.ndarray:
    '''
    Converts a 2D matrix into a 3D matrix so that numbers can be stored as images with high accuracy
    :param layer: weight or bias of layer as a numpy array
    :param normalizer: factor all elements are divided by such that the range remains within 0 and 1
    :return: 3D array of same data, but split along R G B A channels
    '''

    if layer.ndim == 1:  # stupid 1D hack
        layer = np.expand_dims(layer, axis=0)

    lenY, lenX = np.shape(layer)
    output = np.zeros((lenY, lenX, 4)).astype('uint8')

    # Yes I am using for loops. Bite me
    for y in range(lenY):
        for x in range(lenX):
            num = _numToNormalizedColor(layer[y, x], normalizer)
            for c in range(4):
                output[y, x, c] = num[c]

    return output

def _numToNormalizedColor(number: float, normalizer: float) -> np.ndarray:
    '''
    Normalizes and converts number from a float into a RGBA int array
    :param number: float to convert
    :param normalizer: matrix wide normalization factor
    :return: RGBA int array
    '''

    number = (number / normalizer) + 1
    R = int(math.floor(number * 100))
    G = int(math.floor(number * 10000)) - (100*R)
    B = int(math.floor(number * 1000000)) - (10000*R) - (100*G)
    A = int(math.floor(number * 100000000)) - (1000000*R) - (10000*G) - (100*B)
    return np.array([R, G, B, A]).astype('uint8')

def _calculateNormalizer( layer: np.ndarray) -> float:
    '''
    Calculates the factor that entire matrix can be divided by such that the range is within 0 - 1.
    Process can be reversed via (input - 1.0) * normalizer
    :param layer:
    :return:
    '''

    maxVal: float = np.max(layer)
    minVal: float = np.max(layer)

    output: float
    if abs(minVal) > maxVal:
        output = 2.0 * abs(minVal)
    else:
        output = 2.0 * maxVal
    return output