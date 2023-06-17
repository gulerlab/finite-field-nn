# this project
from weight_initializations import kaiming_uniform
from modules import RealModule


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
        self._weight = init_fnc(self._in_dim, self._out_dim)

    def forward(self, input_data):
        self._input_data = input_data
        return self._input_data @ self._weight

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = (self._input_data.T @ propagated_error)

    def optimize(self, learning_rate):
        self._weight = self._weight - learning_rate * self._gradient

    def loss(self):
        return self._propagated_error @ self._weight.T


class RealConvLayer(RealModule):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size

    def forward(self, input_data):
        pass

    def backprop(self, propagated_error):
        pass

    def optimize(self, learning_rate):
        pass

    def loss(self):
        pass
