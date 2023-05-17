import logging
import numpy as np
from numpy import ndarray
import math

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


############
# domain converting operations
############


def to_finite_field_domain(real: ndarray, quantization_bit: int, prime: int) -> ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real)
    finite_field = np.zeros(int_domain.shape, dtype=np.uint64)
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = int_domain[~negative_mask]
    finite_field[negative_mask] = prime - (int_domain[negative_mask] * -1).astype(np.uint64)
    return finite_field


def to_int_domain(real: ndarray, quantization_bit: int) -> ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real).astype(np.int64)
    return int_domain


def to_real_domain(finite_field: ndarray, quantization_bit: int, prime: int) -> ndarray:
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    real_domain = np.zeros(finite_field.shape, dtype=np.float64)
    real_domain[~negative_mask] = finite_field[~negative_mask]
    real_domain[negative_mask] = -1 * (prime - finite_field[negative_mask]).astype(np.float64)
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def from_int_to_real_domain(int_domain: ndarray, quantization_bit: int) -> ndarray:
    real_domain = int_domain.astype(np.float64)
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def int_truncation(int_domain: ndarray, scale_down: int) -> ndarray:
    real_domain = int_domain.astype(np.int64)
    real_domain = real_domain / (2 ** scale_down)
    real_domain_floor = np.floor(real_domain)

    zero_distributions = real_domain - real_domain_floor
    stochastic_fnc = np.vectorize(lambda x: np.random.choice([0, 1], 1, p=[1 - x, x])[0])
    zero_distributions = stochastic_fnc(zero_distributions)

    truncated_int_domain = (real_domain_floor + zero_distributions).astype(np.int64)
    return truncated_int_domain


def from_finite_field_to_int_domain(finite_field: ndarray, prime: int) -> ndarray:
    int_domain = np.zeros(finite_field.shape, dtype=np.int64)
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    int_domain[~negative_mask] = finite_field[~negative_mask]
    int_domain[negative_mask] = -1 * (prime - finite_field[negative_mask]).astype(np.int64)
    return int_domain


def from_int_to_finite_field_domain(int_domain: ndarray, prime: int) -> ndarray:
    finite_field = np.zeros(int_domain.shape, dtype=np.uint64)
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = int_domain[~negative_mask]
    finite_field[negative_mask] = int_domain[negative_mask] + prime
    return finite_field


def finite_field_truncation(finite_field: ndarray, scale_down: int, prime: int) -> ndarray:
    int_domain = from_finite_field_to_int_domain(finite_field, prime)
    int_domain = int_truncation(int_domain, scale_down)
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


def to_real_domain_int(finite_field: int, quantization_bit: int, prime: int) -> ndarray:
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


###################
# transformers
###################


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


class ToNumpy(object):
    def __call__(self, sample):
        return sample.numpy()


#############
# numpy data
#############

def load_all_data(quantization_bit_data: int, quantization_bit_label: int, prime: int):
    # transformations
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = FashionMNIST('./data', train=True, transform=transform, target_transform=target_transform,
                                 download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = FashionMNIST('./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = next(iter(train_loader))
    test_data, test_label = next(iter(test_loader))
    train_data, train_label, test_data, test_label = train_data.squeeze(), train_label.squeeze(), test_data.squeeze(),\
        test_label.squeeze()
    train_data, train_label, test_data, test_label = train_data.numpy(), train_label.numpy(), test_data.numpy(), \
        test_label.numpy()
    train_data, train_label, test_data = to_finite_field_domain(train_data, quantization_bit_data, prime),\
        to_finite_field_domain(train_label, quantization_bit_label, prime),\
        to_finite_field_domain(test_data, quantization_bit_data, prime)
    return train_data, train_label, test_data, test_label


#############
# utils for debug
#############

def info(msg, verbose=True):
    logging.info(msg)
    if verbose:
        print(msg)