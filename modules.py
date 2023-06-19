from abc import ABC, abstractmethod


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
