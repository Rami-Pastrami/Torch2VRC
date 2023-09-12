# Responsible for exporting Pytorch weights as PNGs that can be read in via HLSL shaders
import numpy as np
import math
from PIL import Image as im
from pathlib import Path

def export_np_array_as_png(inputArray: np.ndarray, filePathName: Path) -> float:
    '''
    Saves Layer Numpy Array as a PNG image
    :param inputArray: 2D numpy array from PyTorch itself
    :param filePathName: file name / path to save PNG at
    :param normalizer: value needed to center the array into a 0 - 1 scale
    :return: normalizer value
    '''

    def _numpy_layer_to_rgba_array(layer: np.ndarray, normalizer: float) -> np.ndarray:
        '''
        Converts a 2D matrix into a 3D matrix so that numbers can be stored as images with high accuracy
        :param layer: weight or bias of layer as a numpy array
        :param normalizer: factor all elements are divided by such that the range remains within 0 and 1
        :return: 3D array of same data, but split along R G B A channels
        '''

        def _num_to_normalized_color(number: float, normalizer: float) -> np.ndarray:
            '''
            Normalizes and converts number from a float into a RGBA int array
            :param number: float to convert
            :param normalizer: matrix wide normalization factor
            :return: RGBA int array
            '''

            number = (number / normalizer) + 1
            R = int(math.floor(number * 100))
            G = int(math.floor(number * 10000)) - (100 * R)
            B = int(math.floor(number * 1000000)) - (10000 * R) - (100 * G)
            A = int(math.floor(number * 100000000)) - (1000000 * R) - (10000 * G) - (100 * B)
            return np.array([R, G, B, A]).astype('uint8')

        if layer.ndim == 1:
            [lenX] = np.shape(layer)
            output = np.zeros((1, lenX, 4)).astype('uint8')
            # Yes I am using for loops. Bite me
            for x in range(lenX):
                num = _num_to_normalized_color(layer[x], normalizer)
                for c in range(4):
                    output[0, x, c] = num[c]
            return output

        if layer.ndim == 2:
            lenY, lenX = np.shape(layer)
            output = np.zeros((lenY, lenX, 4)).astype('uint8')

            # Yes I am using for loops. Bite me
            for y in range(lenY):
                for x in range(lenX):
                    num = _num_to_normalized_color(layer[y, x], normalizer)
                    for c in range(4):
                        output[y, x, c] = num[c]
            return output
        # if nothing else
        return np.ndarray([])
    
    normalizer: float = calculate_normalizer(inputArray)
    RGBAArray: np.ndarray = _numpy_layer_to_rgba_array(inputArray, normalizer)
    ImageData = im.fromarray(RGBAArray, mode="RGBA")
    ImageData.save(filePathName)
    return normalizer

def calculate_normalizer(layer: np.ndarray) -> float:
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

def reverse_normalizer(normalized_layer: np.ndarray, normalizer: float) -> np.ndarray:

    return (normalized_layer - 1.0) * normalizer

