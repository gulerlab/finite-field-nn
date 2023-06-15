from real_module import RealModule
from real_layers import RealLinearLayer
from real_activations import RealQuadraticAct


class RealNetworkLinearTest(RealModule):
    def __init__(self):
        super().__init__()
        self.__model = [
            RealLinearLayer(784, 128),
            RealQuadraticAct(),
            RealLinearLayer(128, 10)
        ]

    def forward(self, input_data):
        self._input_data = input_data
        curr_data = self._input_data
        for layer in self.__model:
            curr_data = layer.forward(curr_data)
        return curr_data

    def backprop(self, propagated_error):
        self.__propagated_error = propagated_error
        curr_error = self.__propagated_error
        for layer in reversed(self.__model):
            layer.backprop(curr_error)
            curr_error = layer.loss()

    def optimize(self, learning_rate):
        for layer in self.__model:
            if isinstance(layer, RealModule):
                layer.optimize(learning_rate)

    def loss(self):
        pass
