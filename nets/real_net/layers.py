# this project
from weight_initializations import kaiming_uniform, kaiming_uniform_conv
from modules import RealModule

import numpy as np


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


# TODO: for now it will be just for stride=1 and padding=1, I will consider other setups after applying the main conv
#  operation
class RealConvLayer(RealModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0, 0, 0)):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self.__init_weights()

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    def __init_weights(self, init_fnc=kaiming_uniform_conv):
        self._weight = init_fnc(self._in_channels, self._out_channels, self._kernel_size)

    def __generate_patches(self):
        self.__patches = []
        num_of_samples, _, image_height, image_width = self._input_data.shape
        kernel_height, kernel_width = self._kernel_size
        stride_over_height, stride_over_width = self._stride
        padding_top, padding_bottom, padding_left, padding_right = self._padding
        curr_input_data = np.zeros((self._input_data.shape[0], self._input_data.shape[1], image_height
                                    + (padding_top + padding_bottom), image_width +
                                    (padding_left + padding_right)), dtype=self._input_data.dtype)
        curr_input_data[:, :, padding_top:(padding_top
                                           + image_height), padding_left:(padding_left
                                                                          + image_height)] = self._input_data
        image_height = image_height + (padding_top + padding_bottom)
        image_width = image_width + (padding_top + padding_bottom)
        for width_idx in range(0, image_width - kernel_width + stride_over_width, stride_over_width):
            for height_idx in range(0, image_height - kernel_height + stride_over_height, stride_over_height):
                self.__patches.append(curr_input_data[:, :, height_idx:(height_idx + kernel_height),
                                      width_idx:(width_idx + kernel_width)])

    def forward(self, input_data):
        self._input_data = input_data
        self.__generate_patches()
        pass

    def backprop(self, propagated_error):
        pass

    def optimize(self, learning_rate):
        pass

    def loss(self):
        pass
