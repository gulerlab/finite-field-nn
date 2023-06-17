import copy

import torch
from torch import Tensor
import numpy as np
import math


def to_finite_field_domain(real: Tensor, quantization_bit: int, prime: int) -> Tensor:
    double_real = real.type(torch.double)
    scaled_real = double_real * (2 ** quantization_bit)
    int_domain = torch.round(scaled_real).type(torch.long)
    negative_mask = int_domain < 0
    int_domain[negative_mask] = int_domain[negative_mask] + prime
    return int_domain


def to_int_domain(real: Tensor, quantization_bit: int) -> Tensor:
    double_real = real.type(torch.double)
    scaled_real = double_real * (2 ** quantization_bit)
    int_domain = torch.round(scaled_real)
    return int_domain.type(torch.long)


def to_real_domain(finite_field: Tensor, quantization_bit: int, prime: int) -> Tensor:
    real_domain = finite_field.type(torch.double)
    threshold = (prime - 1) / 2
    negative_mask = real_domain > threshold
    real_domain[negative_mask] = real_domain[negative_mask] - prime
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def from_int_to_real_domain(int_domain: Tensor, quantization_bit: int) -> Tensor:
    real_domain = int_domain.type(torch.double)
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def finite_field_truncation(finite_field: Tensor, scale_down: int) -> Tensor:
    real_domain = finite_field.type(torch.double)
    loop = int(scale_down / 60)
    remainder = scale_down % 60
    for idx in range(loop):
        real_domain = real_domain / (2 ** 60)
    real_domain = real_domain / (2 ** remainder)
    real_domain_floor = torch.floor(real_domain)

    zero_distributions = (real_domain - real_domain_floor).to('cpu')
    zero_distributions.apply_(lambda x: np.random.choice([0, 1], 1, p=[1 - x, x])[0])
    zero_distributions = zero_distributions.to(finite_field.device)

    finite_field_domain = (real_domain_floor + zero_distributions).type(torch.long)
    return finite_field_domain


def from_finite_field_to_int_domain(finite_field: Tensor, prime: int) -> Tensor:
    int_domain = copy.deepcopy(finite_field)
    threshold = (prime - 1) / 2
    negative_mask = int_domain > threshold
    int_domain[negative_mask] = int_domain[negative_mask] - prime
    return int_domain.type(torch.long)


def from_int_to_finite_field_domain(int_domain: Tensor, prime: int) -> Tensor:
    finite_field = int_domain.type(torch.long)
    negative_mask = finite_field < 0
    finite_field[negative_mask] = finite_field[negative_mask] + prime
    return finite_field


def finite_field_truncation_ext(finite_field: Tensor, scale_down: int, prime: int) -> Tensor:
    int_domain = from_finite_field_to_int_domain(finite_field, prime)
    real_domain = int_domain.type(torch.double)
    loop = int(scale_down / 60)
    remainder = scale_down % 60
    for idx in range(loop):
        real_domain = real_domain / (2 ** 60)
    real_domain = real_domain / (2 ** remainder)
    real_domain_floor = torch.floor(real_domain)

    zero_distributions = (real_domain - real_domain_floor).to('cpu')
    zero_distributions.apply_(lambda x: np.random.choice([0, 1], 1, p=[1 - x, x])[0])
    zero_distributions = zero_distributions.to(finite_field.device)

    int_domain = (real_domain_floor + zero_distributions).type(torch.long)
    finite_field_domain = from_int_to_finite_field_domain(int_domain, prime)
    return finite_field_domain


def to_finite_field_domain_int(real: float, quantization_bit: int, prime: int) -> int:
    scaled_real = real * (2 ** quantization_bit)
    finite_field_domain = round(scaled_real)
    if finite_field_domain < 0:
        finite_field_domain = finite_field_domain + prime
    return int(finite_field_domain)


def to_int_domain_int(real: float, quantization_bit: int) -> int:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = round(scaled_real)
    return int(int_domain)


def to_real_domain_int(finite_field: int, quantization_bit: int, prime: int) -> Tensor:
    threshold = (prime - 1) / 2
    real_domain = finite_field
    if real_domain > threshold:
        real_domain = real_domain - prime
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def from_int_to_real_domain_int(int_domain: int, quantization_bit: int):
    real_domain = int_domain / (2 ** quantization_bit)
    return real_domain


def finite_field_truncation_int(finite_field: int, scale_down: int) -> int:
    real_domain = finite_field / (2 ** scale_down)
    real_domain_floor = math.floor(real_domain)
    remainder = real_domain - real_domain_floor
    random_bit = np.random.choice([0, 1], 1, p=[1 - remainder, remainder])[0]
    finite_field_domain = int(real_domain_floor + random_bit)
    return finite_field_domain


def relu(input_sample: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(input_sample.size())
    pos_mask = input_sample > 0
    result[pos_mask] = input_sample[pos_mask]
    return result


def grad_relu(input_sample: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(input_sample.size())
    pos_mask = input_sample > 0
    result[pos_mask] = 1
    return result


class ToFiniteFieldDomain(object):
    def __init__(self, scale_input_parameter, prime):
        self.__scale_input_parameter = scale_input_parameter
        self.__prime = prime

    @property
    def scale_input_parameter(self):
        return self.__scale_input_parameter

    @scale_input_parameter.setter
    def scale_input_parameter(self, value):
        self.__scale_input_parameter = value

    @property
    def prime(self):
        return self.__prime

    @prime.setter
    def prime(self, value):
        self.__prime = value

    def __call__(self, sample):
        return to_finite_field_domain(sample, self.__scale_input_parameter, self.__prime)


class ToIntDomain(object):
    def __init__(self, scale_input_parameter):
        self.__scale_input_parameter = scale_input_parameter

    @property
    def scale_input_parameter(self):
        return self.__scale_input_parameter

    @scale_input_parameter.setter
    def scale_input_parameter(self, value):
        self.__scale_input_parameter = value

    def __call__(self, sample):
        return to_int_domain(sample, self.__scale_input_parameter)
