from modules import RealActivationModule


class RealQuadraticActivation(RealActivationModule):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        self._input_data = input_data
        return self._input_data ** 2

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = 2 * self._input_data

    def loss(self):
        return self._propagated_error * self._gradient
