import numpy as np


class RealMSELoss:
    def __init__(self):
        self.__diff = None

    def forward(self, input_data, ground_truth):
        self.__diff = ground_truth - input_data
        return (np.linalg.norm(self.__diff) ** 2) / self.__diff.shape[0]

    def error_derivative(self):
        return -2 * self.__diff / self.__diff.shape[0]
