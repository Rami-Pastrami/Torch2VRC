import torch as pt
from Torch2VRC.Activation.ActivationBase import ActivationBase

class ActivationNone(ActivationBase):
    """
    Activation function of "none", IE no activation function at all
    """

    def __init__(self):
        super().__init__()
        self.moustache_template = "\n"

    def activation_function(self, to_activate: pt.Tensor) -> pt.Tensor:
        return to_activate

