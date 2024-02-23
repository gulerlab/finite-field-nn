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

class GAPTruncation(ActivationModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self._input_data = input_data
        self.__divisor = self._input_data.shape[2] * self._input_data.shape[3]
        channelwise_sum = np.sum(self._input_data.reshape(self._input_data.shape[0], self._input_data.shape[1], -1), axis=-1)
        channelwise_sum = channelwise_sum / self.__divisor
        return channelwise_sum 

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
           

    def loss(self):
        unscaled_flattened_resulting_error = self._propagated_error[:, :, np.newaxis] @ np.ones((1, self.__divisor))
        flattened_resulting_error = unscaled_flattened_resulting_error / self.__divisor
        resulting_error = flattened_resulting_error.reshape(self._input_data.shape)
        return resulting_error