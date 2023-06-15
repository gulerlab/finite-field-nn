# this project
from real_weight_initialization import kaiming_uniform
from real_module import RealModule


class RealLinearLayer(RealModule):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # protected
        self._in_dim = in_dim
        self._out_dim = out_dim
        self.__init_weights()

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    def __init_weights(self, init_fnc=kaiming_uniform):
        self.__weight = init_fnc(self._in_dim, self._out_dim)

    def forward(self, input_data):
        self._input_data = input_data
        return self._input_data @ self.__weight

    def backprop(self, propagated_error):
        self.__propagated_error = propagated_error
        self.__gradient = self._input_data.T @ propagated_error

    def optimize(self, learning_rate):
        self.__weight = self.__weight - learning_rate * self.__gradient

    def loss(self):
        return self.__propagated_error @ self.__weight.T
