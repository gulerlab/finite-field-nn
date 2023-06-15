from abc import ABC, abstractmethod


class RealModule(ABC):
    def __init__(self):
        super().__init__()
        # protected
        self._input_data = None

        # private
        self.__gradient = None
        self.__weight = None
        self.__propagated_error = None

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


class RealActivation(ABC):
    def __init__(self):
        super().__init__()
        # protected
        self._input_data = None

        # private
        self.__gradient = None
        self.__propagated_error = None

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backprop(self, propagated_error):
        pass

    @abstractmethod
    def loss(self):
        pass
