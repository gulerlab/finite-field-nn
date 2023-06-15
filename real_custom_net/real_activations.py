from real_module import RealActivation


class RealQuadraticAct(RealActivation):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self._input_data = input_data
        return self._input_data ** 2

    def backprop(self, propagated_error):
        self.__propagated_error = propagated_error
        self.__gradient = 2 * self._input_data

    def loss(self):
        return self.__propagated_error * self.__gradient
