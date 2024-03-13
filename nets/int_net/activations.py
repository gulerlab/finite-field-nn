from nets.modules import ActivationModule
import numpy as np
from nets.int_net.utils import int_truncation, update_minmax

class GAPTruncation(ActivationModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self._input_data = input_data
        self.__flat_size = self._input_data.shape[2] * self._input_data.shape[3]
        self.__divisor = int(np.log2(self.__flat_size))
        channelwise_sum = np.sum(self._input_data.reshape(self._input_data.shape[0], self._input_data.shape[1], -1), axis=-1)
        channelwise_sum = int_truncation(channelwise_sum, self.__divisor)
        return channelwise_sum 

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
           
    def loss(self):
        resulting_error = []
        for sample_index in range(self._propagated_error.shape[0]):
            resulting_error.append(self._propagated_error[sample_index, :, np.newaxis] @ np.ones((1, self.__flat_size), dtype=np.int64))
        resulting_error = np.stack(resulting_error)
        resulting_error = int_truncation(resulting_error, self.__divisor)
        resulting_error = resulting_error.reshape(self._input_data.shape)
        return resulting_error
    
class GaloisQuadraticActivation(ActivationModule):
    def __init__(self, quantization_bit):
        super().__init__()
        self.quantization_bit = quantization_bit

    def forward(self, input_data):
        self._input_data = input_data
        out = self._input_data ** 2
        out = int_truncation(out, self.quantization_bit)
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = 2 * self._input_data
        update_minmax(self._gradient)

    def loss(self):
        out = self._propagated_error * self._gradient
        out = int_truncation(out, self.quantization_bit)
        return out