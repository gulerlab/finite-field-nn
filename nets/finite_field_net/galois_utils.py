import numpy as np
import logging
from numpy import ndarray
import galois


############
# domain converting operations
############


def to_finite_field_domain(real: ndarray, quantization_bit: int, prime: int, field) -> galois.Array:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real).astype(int)
    finite_field = np.zeros(int_domain.shape, dtype='object')
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = int_domain[~negative_mask]
    finite_field[negative_mask] = prime + int_domain[negative_mask]
    return field(finite_field)


def to_real_domain(finite_field: galois.Array, quantization_bit: int, prime: int) -> ndarray:
    finite_field = finite_field.view(np.ndarray)
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    real_domain = np.empty(finite_field.shape)
    real_domain[~negative_mask] = finite_field[~negative_mask] / (2 ** quantization_bit) 
    real_domain[negative_mask] = (finite_field[negative_mask] - prime) / (2 ** quantization_bit)
    return real_domain


def int_truncation(int_domain: ndarray, scale_down: int) -> ndarray:
    int_domain = int_domain >> scale_down
    return int_domain


def from_finite_field_to_int_domain(finite_field: galois.Array, prime: int) -> ndarray:
    finite_field = finite_field.view(np.ndarray)
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    finite_field[negative_mask] = finite_field[negative_mask] - prime
    return finite_field


def from_int_to_finite_field_domain(int_domain: ndarray, prime: int, field) -> galois.Array:
    negative_mask = int_domain < 0
    int_domain[negative_mask] = int_domain[negative_mask] + prime
    return field(int_domain)


def finite_field_truncation(finite_field: galois.Array, scale_down: int, prime: int, field) -> galois.Array:
    finite_field = from_finite_field_to_int_domain(finite_field, prime)
    finite_field = int_truncation(finite_field, scale_down)
    finite_field = from_int_to_finite_field_domain(finite_field, prime, field)
    return finite_field


###################
# transformers
###################


class ToFiniteFieldDomain(object):
    def __init__(self, scale_input_parameter, prime, field):
        self.__scale_input_parameter = scale_input_parameter
        self.__prime = prime
        self.__field = field

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

    @property
    def field(self):
        return self.__field

    @field.setter
    def field(self, value):
        self.__field = value

    def __call__(self, sample):
        return to_finite_field_domain(sample, self.__scale_input_parameter, self.__prime, self.__field)


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
# integer representation with object definition
#############

# TODO: with this approach the bias problem (if it affects something) can be solved

# def int_remainder_to_decimal_scalar(remainder, scale_down):
#     bit_str = np.binary_repr(remainder, width=scale_down)
#     result = 0.0
#     for idx, bit_elem in enumerate(bit_str):
#         if bit_elem == '1':
#             result += 2 ** (-1 * (idx + 1))
#     return result

# def int_truncation_object(int_domain: ndarray, scale_down: int) -> ndarray:
#     real_domain_floor = int_domain >> scale_down
#     remainders = int_domain % (2 ** scale_down)
#     zero_distributions_fnc = np.vectorize(lambda x: int_remainder_to_decimal_scalar(x, scale_down))
#     zero_distributions = zero_distributions_fnc(remainders)
#     stochastic_fnc = np.vectorize(lambda x: np.random.choice([0, 1], 1, p=[1 - x, x])[0])
#     zero_distributions = stochastic_fnc(zero_distributions)

#     truncated_int_domain = (real_domain_floor + zero_distributions).astype('object')
#     return truncated_int_domain