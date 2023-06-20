# this project
from weight_initializations import kaiming_uniform, kaiming_uniform_conv
from modules import Module

import numpy as np


class FiniteFieldLinearLayer(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # protected
        self._in_dim = in_dim
        self._out_dim = out_dim
        self.__init_weights()

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    def __init_weights(self, init_fnc=kaiming_uniform):
        self._weight = init_fnc(self._in_dim, self._out_dim)

    def forward(self, input_data):
        self._input_data = input_data
        return self._input_data @ self._weight

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = (self._input_data.T @ propagated_error)

    def optimize(self, learning_rate):
        self._weight = self._weight - learning_rate * self._gradient

    def loss(self):
        return self._propagated_error @ self._weight.T

# class RealPiNetSecondOrderLinearLayer(Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.__first_fc = RealLinearLayer(in_dim, out_dim)
#         self.__second_fc = RealLinearLayer(in_dim, out_dim)
#         self.__inner_forward = {}
#         self.__inner_prop = {}
#
#     def forward(self, input_data):
#         self._input_data = input_data
#         first_out = self.__first_fc.forward(input_data)
#         second_out = self.__second_fc.forward(input_data)
#         self.__inner_forward['out_1'] = first_out
#         self.__inner_forward['out_2'] = second_out
#         return (first_out * second_out) + first_out
#
#     def backprop(self, propagated_error):
#         self._propagated_error = propagated_error
#         second_prop = self._propagated_error * self.__inner_forward['out_1']
#         first_prop = (self._propagated_error * self.__inner_forward['out_2']) + self._propagated_error
#         self.__inner_prop['prop_1'] = first_prop
#         self.__inner_prop['prop_2'] = second_prop
#
#         self.__first_fc.backprop(first_prop)
#         self.__second_fc.backprop(second_prop)
#
#     def optimize(self, learning_rate):
#         self.__first_fc.optimize(learning_rate)
#         self.__second_fc.optimize(learning_rate)
#
#     def loss(self):
#         return self.__first_fc.loss() + self.__second_fc.loss()