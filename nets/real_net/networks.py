from modules import Module
from layers import RealLinearLayer, RealConvLayer
from activations import RealQuadraticActivation, RealReLU
from utils import Flatten


# TODO: after conv implementation is done implement some known networks
# TODO: delete the previous implementations after implementing the known ones
#  and make it customizable
class RealNetworkLinearTest(Module):
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
            if isinstance(layer, Module):
                layer.optimize(learning_rate)

    def loss(self):
        pass


class RealNetworkLinearTest_v2(Module):
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
            if isinstance(layer, Module):
                layer.optimize(learning_rate)

    def loss(self):
        pass


class RealLeNet(Module):
    def __init__(self):
        super().__init__()
        self.__model = [
            RealConvLayer(1, 6, (5, 5), padding=(2, 2, 2, 2)),
            RealQuadraticActivation(),
            RealConvLayer(6, 16, (5, 5)),
            RealQuadraticActivation(),
            Flatten(),
            RealLinearLayer(9216, 120),
            RealQuadraticActivation(),
            RealLinearLayer(120, 84),
            RealQuadraticActivation(),
            RealLinearLayer(84, 10)
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
            if isinstance(layer, Module):
                layer.optimize(learning_rate)

    def loss(self):
        pass


class RealLeNetReLU(Module):
    def __init__(self):
        super().__init__()
        self.__model = [
            RealConvLayer(1, 6, (5, 5), padding=(2, 2, 2, 2)),
            RealReLU(),
            RealConvLayer(6, 16, (5, 5)),
            RealReLU(),
            Flatten(),
            RealLinearLayer(9216, 120),
            RealReLU(),
            RealLinearLayer(120, 84),
            RealReLU(),
            RealLinearLayer(84, 10)
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
            if isinstance(layer, Module):
                layer.optimize(learning_rate)

    def loss(self):
        pass
