import numpy as np
from modules import ActivationModule


class Flatten(ActivationModule):
    def __init__(self):
        super().__init__()
        self._input_data_shape = None

    def forward(self, input_data):
        self._input_data_shape = input_data.shape
        return np.reshape(input_data, (self._input_data_shape[0], -1))

    def backprop(self, propagated_error):
        self._gradient = np.reshape(propagated_error, self._input_data_shape)

    def loss(self):
        return self._gradient
