from abc import ABC, abstractmethod
import numpy as np


class Module(ABC):
    def __init__(self):
        super().__init__()
        # protected
        self._input_data = None
        self._gradient = None
        self._weight = None
        self._propagated_error = None

    @property
    def input_data(self):
        return self._input_data

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backprop(self, propagated_error):
        pass

    @abstractmethod
    def optimize(self, learning_rate):
        pass

    @abstractmethod
    def loss(self):
        pass


class Network(Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    @property
    def model(self):
        return self._model

    def forward(self, input_data):
        self._input_data = input_data
        curr_data = self._input_data
        for layer in self._model:
            curr_data = layer.forward(curr_data)
        return curr_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        curr_error = self._propagated_error
        for layer in reversed(self._model):
            layer.backprop(curr_error)
            curr_error = layer.loss()

    def optimize(self, learning_rate):
        for layer in self._model:
            if isinstance(layer, Module):
                layer.optimize(learning_rate)

    def loss(self):
        pass


class ActivationModule(ABC):
    def __init__(self):
        super().__init__()
        # protected
        self._input_data = None
        self._gradient = None
        self._propagated_error = None

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backprop(self, propagated_error):
        pass

    @abstractmethod
    def loss(self):
        pass


class Flatten(ActivationModule):
    def __init__(self):
        super().__init__()
        self._input_data_shape = None

    def forward(self, input_data):
        self._input_data_shape = input_data.shape
        return np.reshape(input_data, (self._input_data_shape[0], -1))

    def backprop(self, propagated_error):
        self._gradient = np.reshape(propagated_error, self._input_data_shape)

    def loss(self):
        return self._gradient
