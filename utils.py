import numpy as np
import logging
from numpy import ndarray
import math
from itertools import product
from numba import jit


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


# noinspection DuplicatedCode
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
# common dataset operations
#############
def create_batch_data(train_data, train_label, test_data, test_label, batch_size):
    train_num_samples, test_num_samples = train_data.shape[0], test_data.shape[0]
    number_of_full_batch_train = int(train_num_samples / batch_size)
    last_batch_size_train = train_num_samples % batch_size

    number_of_full_batch_test = int(test_num_samples / batch_size)
    last_batch_size_test = test_num_samples % batch_size

    last_batch_train_data = None
    if last_batch_size_train != 0:
        last_batch_train_data = train_data[train_num_samples - last_batch_size_train:, :]

    if number_of_full_batch_train > 0:
        train_data = np.split(train_data[:train_num_samples - last_batch_size_train, :], number_of_full_batch_train)
    else:
        train_data = []
    if last_batch_train_data is not None:
        train_data.append(last_batch_train_data)

    last_batch_train_label = None
    if last_batch_size_train != 0:
        last_batch_train_label = train_label[train_num_samples - last_batch_size_train:, :]

    if number_of_full_batch_train > 0:
        train_label = np.split(train_label[:train_num_samples - last_batch_size_train, :], number_of_full_batch_train)
    else:
        train_label = []
    if last_batch_train_label is not None:
        train_label.append(last_batch_train_label)

    last_batch_test_data = None
    if last_batch_size_test != 0:
        last_batch_test_data = test_data[test_num_samples - last_batch_size_test:, :]

    if number_of_full_batch_test > 0:
        test_data = np.split(test_data[:test_num_samples - last_batch_size_test, :], number_of_full_batch_test)
    else:
        test_data = []
    if last_batch_test_data is not None:
        test_data.append(last_batch_test_data)

    last_batch_test_label = None
    if last_batch_size_test != 0:
        last_batch_test_label = test_label[test_num_samples - last_batch_size_test:]

    if number_of_full_batch_train > 0:
        test_label = np.split(test_label[:test_num_samples - last_batch_size_test], number_of_full_batch_test)
    else:
        test_label = []
    if last_batch_test_label is not None:
        test_label.append(last_batch_test_label)

    return train_data, train_label, test_data, test_label


#############
# utils for debug
#############

def info(msg, verbose=True):
    logging.info(msg)
    if verbose:
        print(msg)


#############
# galois field implementations
#############

def to_galois_field(real, quantization_bit, prime, field):
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real)
    galois_field = field.Zeros(int_domain.shape)
    negative_mask = int_domain < 0
    galois_field[~negative_mask] = int_domain[~negative_mask].astype(np.uint64)
    galois_field[negative_mask] = prime - (int_domain[negative_mask] * -1).astype(np.uint64)
    return galois_field


def from_galois_to_real_domain(galois_field, quantization_bit, prime):
    threshold = (prime - 1) / 2
    negative_mask = galois_field > threshold
    real_domain = np.zeros(galois_field.shape, dtype=np.float64)
    positive_galois_field = galois_field[~negative_mask]
    positive_items = [x.item() for x in positive_galois_field]
    negative_galois_field = galois_field[negative_mask]
    negative_items = [x.item() for x in negative_galois_field]
    real_domain[~negative_mask] = np.asarray(positive_items)
    real_domain[negative_mask] = np.asarray(negative_items) - prime
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def from_galois_to_finite_field(galois_field):
    shape = galois_field.shape
    finite_field = np.zeros(shape, dtype=np.uint64)
    shape_prod = [range(shape_idx) for shape_idx in shape]
    generator = product(*shape_prod)
    for idx in generator:
        finite_field[idx] = galois_field[idx].item()
    return finite_field


#############
# integer representation with object definition
#############

def int_remainder_to_decimal_scalar(remainder, scale_down):
    bit_str = np.binary_repr(remainder, width=scale_down)
    result = 0.0
    for idx, bit_elem in enumerate(bit_str):
        if bit_elem == '1':
            result += 2 ** (-1 * (idx + 1))
    return result


def to_int_domain_object(real: ndarray, quantization_bit: int) -> ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real).astype(np.int64)
    int_domain = int_domain.astype('object')
    return int_domain


def int_truncation_object(int_domain: ndarray, scale_down: int) -> ndarray:
    real_domain_floor = int_domain >> scale_down
    remainders = int_domain % (2 ** scale_down)
    zero_distributions_fnc = np.vectorize(lambda x: int_remainder_to_decimal_scalar(x, scale_down))
    zero_distributions = zero_distributions_fnc(remainders)
    stochastic_fnc = np.vectorize(lambda x: np.random.choice([0, 1], 1, p=[1 - x, x])[0])
    zero_distributions = stochastic_fnc(zero_distributions)

    truncated_int_domain = (real_domain_floor + zero_distributions).astype('object')
    return truncated_int_domain


def jit_int_truncation_object(int_domain: ndarray, scale_down: int) -> ndarray:
    real_domain_floor = int_domain >> scale_down
    remainders = int_domain % (2 ** scale_down)
    zero_distributions_fnc = jit(lambda x: int_remainder_to_decimal_scalar(x, scale_down))
    zero_distributions = zero_distributions_fnc(remainders)
    stochastic_fnc = jit(lambda x: np.random.choice([0, 1], 1, p=[1 - x, x])[0])
    zero_distributions = stochastic_fnc(zero_distributions)

    truncated_int_domain = (real_domain_floor + zero_distributions).astype('object')
    return truncated_int_domain


#############
# finite field representation with object definition
#############

def to_finite_field_domain_object(real, quantization_bit, prime):
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real)
    finite_field = np.zeros(int_domain.shape, dtype='object')
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = (int_domain[~negative_mask]).astype(np.uint64)
    finite_field[negative_mask] = prime - (int_domain[negative_mask] * -1).astype(np.uint64)
    return finite_field


def finite_field_truncation_object(finite_field, scale_down, prime):
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    neg_to_pos = (-1 * finite_field[negative_mask]) % prime
    neg_to_pos = int_truncation_object(neg_to_pos, scale_down)
    neg_to_pos = (-1 * neg_to_pos) % prime
    pos = finite_field[~negative_mask]
    pos = int_truncation_object(pos, scale_down)

    truncated_finite_field_domain = np.zeros(finite_field.shape, dtype='object')
    truncated_finite_field_domain[negative_mask] = neg_to_pos
    truncated_finite_field_domain[~negative_mask] = pos
    return truncated_finite_field_domain
