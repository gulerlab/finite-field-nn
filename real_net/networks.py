from modules import RealModule
from layers import RealLinearLayer
from activations import RealQuadraticActivation


# TODO: after conv implementation is done implement some known networks
class RealNetworkLinearTest(RealModule):
    def __init__(self):
        super().__init__()
        self.__model = [
            RealLinearLayer(784, 128),
            RealQuadraticActivation(),
            RealLinearLayer(128, 10)
        ]

    def forward(self, input_data):
        self._input_data = input_data
        curr_data = self._input_data
        for layer in self.__model:
            curr_data = layer.forward(curr_data)
        return curr_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        curr_error = self._propagated_error
        for layer in reversed(self.__model):
            layer.backprop(curr_error)
            curr_error = layer.loss()

    def optimize(self, learning_rate):
        for layer in self.__model:
            if isinstance(layer, RealModule):
                layer.optimize(learning_rate)

    def loss(self):
        pass


class RealNetworkLinearTest_v2(RealModule):
    def __init__(self):
        super().__init__()
        self.__model = [
            RealLinearLayer(784, 512),
            RealQuadraticActivation(),
            RealLinearLayer(512, 256),
            RealQuadraticActivation(),
            RealLinearLayer(256, 128),
            RealQuadraticActivation(),
            RealLinearLayer(128, 64),
            RealQuadraticActivation(),
            RealLinearLayer(64, 32),
            RealQuadraticActivation(),
            RealLinearLayer(32, 10),
        ]

    def forward(self, input_data):
        self._input_data = input_data
        curr_data = self._input_data
        for layer in self.__model:
            curr_data = layer.forward(curr_data)
        return curr_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        curr_error = self._propagated_error
        for layer in reversed(self.__model):
            layer.backprop(curr_error)
            curr_error = layer.loss()

    def optimize(self, learning_rate):
        for layer in self.__model:
            if isinstance(layer, RealModule):
                layer.optimize(learning_rate)

    def loss(self):
        pass
