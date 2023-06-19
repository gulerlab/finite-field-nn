from modules import ActivationModule
import numpy as np


class RealQuadraticActivation(ActivationModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self._input_data = input_data
        return self._input_data ** 2

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = 2 * self._input_data

    def loss(self):
        return self._propagated_error * self._gradient


class RealReLU(ActivationModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self._input_data = input_data
        out = np.zeros(self._input_data.shape)
        pos_mask = self._input_data > 0
        out[pos_mask] = self._input_data[pos_mask]
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = np.zeros(self._propagated_error.shape)
        pos_mask = self._input_data > 0
        self._gradient[pos_mask] = 1

    def loss(self):
        return self._propagated_error * self._gradient
