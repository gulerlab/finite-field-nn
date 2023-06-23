import numpy as np
from utils import from_int_to_real_domain, int_truncation


class IntegerMSELoss:
    def __init__(self, quantization_label, quantization_batch_size):
        self.__diff = None
        self._quantization_label = quantization_label
        self._quantization_batch_size = quantization_batch_size

    @property
    def quantization_label(self):
        return self._quantization_label

    @property
    def quantization_batch_size(self):
        return self._quantization_batch_size

    def forward(self, input_data, ground_truth):
        self.__diff = ground_truth - input_data
        real_ground_truth = from_int_to_real_domain(ground_truth, self._quantization_label)
        real_input_data = from_int_to_real_domain(input_data, self._quantization_label)
        real_diff = real_ground_truth - real_input_data
        return (np.linalg.norm(real_diff) ** 2) / real_diff.shape[0]

    def error_derivative(self):
        return int_truncation(-2 * self.__diff, self._quantization_batch_size)
