
class AbstractConnectionHelper:
    def __init__(self, connection_name_from_torch: str, target_layer_name: str):
        self.connection_name_from_torch: str = connection_name_from_torch
        self.target_layer_name: str = target_layer_name


class LinearConnectionHelper(AbstractConnectionHelper):
    def __init__(self, connection_name_from_torch: str, target_layer_name: str, source_layer_name: str):
        super().__init__(connection_name_from_torch,target_layer_name)
        self.source_layer_name: str = source_layer_name
