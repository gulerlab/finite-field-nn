import numpy as np
from nets.int_net.utils import int_truncation, to_real_domain, update_minmax


class IntMSELoss:
    def __init__(self, quantization_bit_prediction, batch_size_param):
        self.__diff = None
        self._batch_size_param = batch_size_param
        self._quantization_bit_prediction = quantization_bit_prediction

    @property
    def batch_size_param(self):
        return self._batch_size_param

    @property
    def quantization_bit_prediction(self):
        return self._quantization_bit_prediction

    def forward(self, input_data, ground_truth):
        # common error term calculation
        self.__diff = ground_truth - input_data
        update_minmax(self.__diff)

        real_ground_truth = to_real_domain(ground_truth, self._quantization_bit_prediction)
        real_input_data = to_real_domain(input_data, self._quantization_bit_prediction)
        real_diff = real_ground_truth - real_input_data
        return (np.linalg.norm(real_diff) ** 2) / real_diff.shape[0]

    def error_derivative(self):
        unscaled_error = -2 * self.__diff
        error = int_truncation(unscaled_error, self._batch_size_param)
        return error
