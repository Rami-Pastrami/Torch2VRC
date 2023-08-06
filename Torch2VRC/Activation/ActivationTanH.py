import torch as pt
from Torch2VRC.Activation.ActivationBase import ActivationBase

class ActivationTanH(ActivationBase):
    """
    TabH Activation Function
    """
    def __init__(self, input_array_name: str = "output"):
        super().__init__()
        self.moustache_template = f"output = Activation_Tanh({{input_array_name}})"

    def activation_function(self, to_activate: pt.Tensor) -> pt.Tensor:
        return pt.tan(to_activate)

