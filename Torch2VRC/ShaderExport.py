import numpy as np





class NNRect():

    width: int
    height: int
    centerPos: np.ndarray
    corners: list[np.ndarray]
    isRotated: bool
    originalData: np.ndarray



    def __init__(self, layerData: np.ndarray, XYOffset: np.ndarray):

        self.height, self.width = layerData.shape
        self.centerPos = np.asarray([self.width / 2.0, self.height / 2.0]) + XYOffset  # initial
        self.originalData = layerData

    def get_area(self) -> int:
        return self.width * self.height

    area: int = property(get_area)









# Calc minimum size CRT block per component
# size CRT and organize locations
# Generate var names
# Generate blocks and passes
# Combine on moustache file
# Save Shader
# Save CRT asset file using moustache