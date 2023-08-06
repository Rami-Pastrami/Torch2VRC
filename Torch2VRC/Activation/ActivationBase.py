import torch as pt


class ActivationBase:
    """
    Base Activation Class with required internals
    """
    moustache_template: str

    def __init__(self):
        pass

    def activation_function(self, to_activate: pt.Tensor) -> pt.Tensor:
        pass
