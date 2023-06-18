import numpy as np


def kaiming_uniform(fan_in, fan_out, mode='fan_in'):
    bound = None
    if mode == 'fan_in':
        bound = np.sqrt(1 / fan_in)
    elif mode == 'fan_out':
        bound = np.sqrt(1 / fan_out)

    assert bound is not None, 'wrong mode param'
    return np.random.uniform(low=(-1 * bound), high=bound, size=(fan_in, fan_out))


def kaiming_uniform_conv(fan_in, fan_out, kernel_size, mode='fan_in'):
    bound = None
    if mode == 'fan_in':
        bound = fan_in
    elif mode == 'fan_out':
        bound = fan_out

    kernel_height, kernel_weight = kernel_size
    bound = bound * kernel_height * kernel_height
    bound = np.sqrt(1 / bound)

    assert bound is not None, 'wrong mode param'
    return np.random.uniform(low=(-1 * bound), high=bound, size=(fan_in, kernel_height, kernel_weight, fan_out))
