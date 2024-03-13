import numpy as np

def init_globals(number_of_epochs):
    global min_value
    global max_value
    global epoch
    epoch = 0
    max_int = np.iinfo(np.int64).max
    min_int = np.iinfo(np.int64).min
    min_value = [max_int for _ in range(number_of_epochs)]
    max_value = [min_int for _ in range(number_of_epochs)]