# this project
from weight_initializations import kaiming_uniform
from modules import Module
from utils import to_finite_field_domain, finite_field_truncation

import numpy as np


class FiniteFieldLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_bit_weight, prime, first_layer=False, quantization_bit_input=None):
        super().__init__()
        # protected
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._quantization_bit_weight = quantization_bit_weight
        self._prime = prime
        self._first_layer = first_layer
        if self._first_layer:
            assert quantization_bit_input is not None, 'if the layer is the first layer then the input' \
                                                       ' quantization bit required'
            self._quantization_bit_input = quantization_bit_input
        self.__init_weights()

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def quantization_bit_weight(self):
        return self._quantization_bit_weight

    @property
    def prime(self):
        return self._prime

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_bit_input(self):
        return self._quantization_bit_input

    def __init_weights(self, init_fnc=kaiming_uniform):
        self._weight = init_fnc(self._in_dim, self._out_dim)
        self._weight = to_finite_field_domain(self._weight, self._quantization_bit_weight, self._prime)

    def forward(self, input_data):
        self._input_data = input_data
        out = (self._input_data @ self._weight) % self._prime
        if self._first_layer:
            out = finite_field_truncation(out, self._quantization_bit_input, self._prime)
        else:
            out = finite_field_truncation(out, self._quantization_bit_weight, self._prime)
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_gradient = (self._input_data.T @ propagated_error) % self._prime
        if self._first_layer:
            self._gradient = finite_field_truncation(unscaled_gradient, self._quantization_bit_input, self._prime)
        else:
            self._gradient = finite_field_truncation(unscaled_gradient, self._quantization_bit_weight, self._prime)

    def optimize(self, learning_rate):
        scaled_gradient = finite_field_truncation(self._gradient, learning_rate, self._prime)
        weight_mask = self._weight < scaled_gradient
        weight_diff_grad = np.zeros(self._weight.shape, dtype=np.uint64)
        weight_diff_grad[weight_mask] = (-1 * (scaled_gradient[weight_mask] -
                                               self._weight[weight_mask]).astype(np.int64)) % self._prime
        weight_diff_grad[~weight_mask] = self._weight[~weight_mask] - scaled_gradient[~weight_mask]
        self._weight = weight_diff_grad

    def loss(self):
        return finite_field_truncation((self._propagated_error @ self._weight.T) % self._prime,
                                       self._quantization_bit_weight, self._prime)


class FiniteFieldPiNetSecondOrderLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_bit_weight, prime, first_layer=False, quantization_bit_input=None):
        super().__init__()
        self.__first_fc = FiniteFieldLinearLayer(in_dim, out_dim, quantization_bit_weight, prime,
                                                 first_layer=first_layer, quantization_bit_input=quantization_bit_input)
        self.__second_fc = FiniteFieldLinearLayer(in_dim, out_dim, quantization_bit_weight, prime,
                                                  first_layer=first_layer,
                                                  quantization_bit_input=quantization_bit_input)
        self._prime = prime
        self._quantization_bit_weight = quantization_bit_weight
        self.__inner_forward = {}
        self.__inner_prop = {}

    @property
    def quantization_bit_weight(self):
        return self._quantization_bit_weight

    @property
    def prime(self):
        return self._prime

    def forward(self, input_data):
        self._input_data = input_data
        first_out = self.__first_fc.forward(input_data)
        second_out = self.__second_fc.forward(input_data)
        self.__inner_forward['out_1'] = first_out
        self.__inner_forward['out_2'] = second_out
        unscaled_second_order = (first_out * second_out) % self._prime
        second_order = finite_field_truncation(unscaled_second_order, self._quantization_bit_weight, self._prime)
        out = (second_order + first_out) % self._prime
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_second_prop = (self._propagated_error * self.__inner_forward['out_1']) % self._prime
        second_prop = finite_field_truncation(unscaled_second_prop, self._quantization_bit_weight, self._prime)
        unscaled_first_prop_second_order = (self._propagated_error * self.__inner_forward['out_2']) % self._prime
        first_prop_second_order = finite_field_truncation(unscaled_first_prop_second_order,
                                                          self._quantization_bit_weight, self._prime)
        first_prop = (first_prop_second_order + self._propagated_error) % self._prime
        self.__inner_prop['prop_1'] = first_prop
        self.__inner_prop['prop_2'] = second_prop

        self.__first_fc.backprop(first_prop)
        self.__second_fc.backprop(second_prop)

    def optimize(self, learning_rate):
        self.__first_fc.optimize(learning_rate)
        self.__second_fc.optimize(learning_rate)

    def loss(self):
        return (self.__first_fc.loss() + self.__second_fc.loss()) % self._prime
