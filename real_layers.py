import numpy as np
from abc import ABC, abstractmethod

# this project
from weight_initialization import kaiming_uniform

class RealLayer(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backprop(self):
        pass


class RealLinearLayer:
    def __init__(self, in_dim, out_dim):
        self._in_dim = in_dim
        self._out_dim = out_dim

    def __init_weights(self):
        self.weight = kaiming_uniform(self._in_dim, self._out_dim)

    def forward(self, input_data):
        return input_data @ self.weight

    def backprop(self, propagated_error):
        pass