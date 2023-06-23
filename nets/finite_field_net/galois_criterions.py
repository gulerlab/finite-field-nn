import numpy as np
from utils import finite_field_truncation, from_galois_to_real_domain, from_galois_to_finite_field


class GaloisFieldMSELoss:
    def __init__(self, prime, quantization_bit_prediction, batch_size_param, field):
        self.__diff = None
        self._prime = prime
        self._batch_size_param = batch_size_param
        self._quantization_bit_prediction = quantization_bit_prediction
        self._field = field

    @property
    def prime(self):
        return self._prime

    @property
    def field(self):
        return self._field

    @property
    def batch_size_param(self):
        return self._batch_size_param

    @property
    def quantization_bit_prediction(self):
        return self._quantization_bit_prediction

    def forward(self, input_data, ground_truth):
        # common error term calculation
        self.__diff = ground_truth - input_data

        real_ground_truth = from_galois_to_real_domain(ground_truth, self._quantization_bit_prediction, self._prime)
        real_input_data = from_galois_to_real_domain(input_data, self._quantization_bit_prediction, self._prime)
        real_diff = real_ground_truth - real_input_data
        return (np.linalg.norm(real_diff) ** 2) / real_diff.shape[0]

    def error_derivative(self):
        unscaled_error = -2 * self.__diff
        unscaled_error = from_galois_to_finite_field(unscaled_error)
        error = finite_field_truncation(unscaled_error, self._batch_size_param, self._prime)
        error = self._field(error)
        return error
