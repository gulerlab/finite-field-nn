import numpy as np

def init_globals(number_of_epochs):
    global min_value
    global max_value
    global epoch
    global overflow_monitor
    global prime
    epoch = 0
    overflow_monitor = 0
    prime = -1
    max_int = np.iinfo(np.int64).max
    min_int = np.iinfo(np.int64).min
    min_value = [max_int for _ in range(number_of_epochs)]
    max_value = [min_int for _ in range(number_of_epochs)]
