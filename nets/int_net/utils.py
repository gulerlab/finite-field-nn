import numpy as np
from numpy import ndarray
import nets.int_net.minmax_cache as minmax_cache
import logging

############
# update minmax_cache
############
def update_minmax(int_domain: np.ndarray):
    inner_min_value = int_domain.min()
    inner_max_value = int_domain.max()
    current_epoch = minmax_cache.epoch
    if minmax_cache.min_value[current_epoch] > inner_min_value:
        minmax_cache.min_value[current_epoch] = inner_min_value
    if minmax_cache.max_value[current_epoch] < inner_max_value:
        minmax_cache.max_value[current_epoch] = inner_max_value
    logging.debug('epoch: {}, min cache: {}, max cache: {}'.format(current_epoch, minmax_cache.min_value[current_epoch], minmax_cache.max_value[current_epoch]))

############
# domain converting operations
############

def to_int_domain(real: ndarray, quantization_bit: int) -> np.ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real).astype(np.int64)
    update_minmax(int_domain)
    return int_domain

def to_real_domain(int_domain: np.ndarray, quantization_bit: int) -> ndarray:
    real_domain = int_domain / (2 ** quantization_bit)
    return real_domain

def int_truncation(int_domain: ndarray, scale_down: int) -> ndarray:
    update_minmax(int_domain)
    int_domain = int_domain >> np.int64(scale_down)
    update_minmax(int_domain)
    return int_domain   

###################
# transformers
###################

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