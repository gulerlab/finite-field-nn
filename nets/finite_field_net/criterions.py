import numpy as np
from utils import finite_field_truncation, to_real_domain


class FiniteFieldMSELoss:
    def __init__(self, prime, quantization_bit_prediction, batch_size_param):
        self.__diff = None
        self._prime = prime
        self._batch_size_param = batch_size_param
        self._quantization_bit_prediction = quantization_bit_prediction

    @property
    def prime(self):
        return self._prime

    @property
    def batch_size_param(self):
        return self._batch_size_param

    @property
    def quantization_bit_prediction(self):
        return self._quantization_bit_prediction

    def forward(self, input_data, ground_truth):
        # common error term calculation - finite field subtraction of error
        ground_truth_mask = ground_truth < input_data
        ground_truth_diff_input_data = np.zeros(ground_truth.shape, dtype=np.uint64)
        ground_truth_diff_input_data[ground_truth_mask] = (-1 * (input_data[ground_truth_mask] -
                                                                 ground_truth[ground_truth_mask]).
                                                           astype(np.int64)) % self._prime
        ground_truth_diff_input_data[~ground_truth_mask] = (ground_truth[~ground_truth_mask] -
                                                            input_data[~ground_truth_mask])
        self.__diff = ground_truth_diff_input_data

        real_ground_truth = to_real_domain(ground_truth, self._quantization_bit_prediction, self._prime)
        real_input_data = to_real_domain(input_data, self._quantization_bit_prediction, self._prime)
        real_diff = real_ground_truth - real_input_data
        return (np.linalg.norm(real_diff) ** 2) / real_diff.shape[0]

    def error_derivative(self):
        unscaled_error = ((-2 % self._prime) * self.__diff) % self._prime
        error = finite_field_truncation(unscaled_error, self._batch_size_param, self._prime)
        return error
