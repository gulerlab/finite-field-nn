from modules import Module
from layers import FiniteFieldLinearLayer, FiniteFieldPiNetSecondOrderLinearLayer, FiniteFieldPiNetSecondOrderConvLayer
from utils import Flatten


class FiniteFieldPiNetNetworkLinear(Module):
    def __init__(self, quantization_bit_weight, prime, quantization_bit_input):
        super().__init__()
        self.__model = [
            FiniteFieldPiNetSecondOrderLinearLayer(784, 128, quantization_bit_weight, prime, first_layer=True,
                                                   quantization_bit_input=quantization_bit_input),
            FiniteFieldLinearLayer(128, 10, quantization_bit_weight, prime)
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


class FiniteFieldPiNetNetworkLeNet(Module):
    def __init__(self, quantization_bit_weight, prime, quantization_bit_input):
        super().__init__()
        self.__model = [
            FiniteFieldPiNetSecondOrderConvLayer(1, 6, (5, 5), quantization_bit_weight, prime, padding=(2, 2, 2, 2),
                                                 first_layer=True, quantization_bit_input=quantization_bit_input),
            FiniteFieldPiNetSecondOrderConvLayer(6, 16, (5, 5), quantization_bit_weight, prime),
            Flatten(),
            FiniteFieldPiNetSecondOrderLinearLayer(9216, 120, quantization_bit_weight, prime),
            FiniteFieldPiNetSecondOrderLinearLayer(120, 84, quantization_bit_weight, prime),
            FiniteFieldLinearLayer(84, 10, quantization_bit_weight, prime)
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


class FiniteFieldPiNetNetworkLeNetCIFAR10(Module):
    def __init__(self, quantization_bit_weight, prime, quantization_bit_input):
        super().__init__()
        self.__model = [
            FiniteFieldPiNetSecondOrderConvLayer(3, 6, (5, 5), quantization_bit_weight, prime, padding=(2, 2, 2, 2),
                                                 first_layer=True, quantization_bit_input=quantization_bit_input),
            FiniteFieldPiNetSecondOrderConvLayer(6, 16, (5, 5), quantization_bit_weight, prime),
            Flatten(),
            FiniteFieldPiNetSecondOrderLinearLayer(12544, 120, quantization_bit_weight, prime),
            FiniteFieldPiNetSecondOrderLinearLayer(120, 84, quantization_bit_weight, prime),
            FiniteFieldLinearLayer(84, 10, quantization_bit_weight, prime)
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
