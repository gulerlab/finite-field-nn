import numpy as np


def kaiming_uniform(fan_in, fan_out, mode='fan_in'):
    bound = None
    if mode == 'fan_in':
        bound = np.sqrt(1 / fan_in)
    elif mode == 'fan_out':
        bound = np.sqrt(1 / fan_out)

    assert bound is not None, 'wrong mode param'
    return np.random.uniform(low=(-1 * bound), high=bound, size=(fan_in, fan_out))
