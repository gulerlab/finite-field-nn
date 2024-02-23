from modules import ActivationModule
import numpy as np
from utils import finite_field_truncation, from_galois_to_finite_field
import galois

class GAPTruncation(ActivationModule):
    def __init__(self, prime, field):
        super().__init__()
        self._prime = prime
        self._field = field

    def forward(self, input_data):
        self._input_data = input_data
        self.__flat_size = self._input_data.shape[2] * self._input_data.shape[3]
        self.__divisor = int(np.log2(self.__flat_size))
        channelwise_sum = np.sum(self._input_data.reshape(self._input_data.shape[0], self._input_data.shape[1], -1), axis=-1)
        channelwise_sum = from_galois_to_finite_field(channelwise_sum)
        channelwise_sum = finite_field_truncation(channelwise_sum, self.__divisor, self._prime)
        channelwise_sum = self._field(channelwise_sum)
        return channelwise_sum 

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
           
    def loss(self):
        resulting_error = []
        for sample_index in range(self._propagated_error.shape[0]):
            resulting_error.append(self._propagated_error[sample_index, :, np.newaxis] @ self._field.Ones((1, self.__flat_size)))
        resulting_error = np.stack(resulting_error)
        resulting_error = from_galois_to_finite_field(resulting_error)
        resulting_error = finite_field_truncation(resulting_error, self.__divisor, self._prime)
        resulting_error = self._field(resulting_error)
        resulting_error = resulting_error.reshape(self._input_data.shape)
        return resulting_error