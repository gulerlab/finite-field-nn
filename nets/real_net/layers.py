# this project
from weight_initializations import kaiming_uniform, kaiming_uniform_conv
from modules import Module

import numpy as np


class RealLinearLayer(Module):
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


class RealConvLayer(Module):
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
        for output_width_idx, width_idx in enumerate(range(0, image_width - kernel_width + stride_over_width,
                                                           stride_over_width)):
            for output_height_idx, height_idx in enumerate(
                    range(0, image_height - kernel_height + stride_over_height, stride_over_height)):
                self.__patches.append((
                    output_height_idx, output_width_idx,
                    curr_input_data[:, :, height_idx:(height_idx + kernel_height), width_idx:(width_idx + kernel_width)]
                ))

    def forward(self, input_data):
        self._input_data = input_data
        self.__generate_patches()
        num_of_samples, _, image_height, image_width = self._input_data.shape
        kernel_height, kernel_width = self._kernel_size
        stride_over_height, stride_over_width = self._stride
        padding_top, padding_bottom, padding_left, padding_right = self._padding
        image_height = image_height + (padding_top + padding_bottom)
        image_width = image_width + (padding_top + padding_bottom)
        output_height, output_width = (int(image_height - kernel_height + stride_over_height / stride_over_height),
                                       int(image_width - kernel_width + stride_over_width / stride_over_width))
        output_data = np.empty((num_of_samples, self._out_channels, output_height, output_width),
                               dtype=self._input_data.dtype)
        for patch in self.__patches:
            output_height_idx, output_width_idx, patch_data = patch
            output_data[:, :, output_height_idx, output_width_idx] = (np.reshape(patch_data, (num_of_samples, -1)) @
                                                                      np.reshape(self._weight, (-1,
                                                                                                self._out_channels)))
        return output_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = np.zeros(self._weight.shape, dtype=self._weight.dtype)
        for patch in self.__patches:
            output_height_idx, output_width_idx, patch_data = patch
            self._gradient += (np.reshape(patch_data, (patch_data.shape[0], -1)).T
                               @ self._propagated_error[:, :, output_height_idx, output_width_idx])\
                .reshape(self._weight.shape)

    def optimize(self, learning_rate):
        self._weight = self._weight - learning_rate * self._gradient

    def loss(self):
        num_of_samples, _, image_height, image_width = self._input_data.shape
        kernel_height, kernel_width = self._kernel_size
        stride_over_height, stride_over_width = self._stride
        padding_top, padding_bottom, padding_left, padding_right = self._padding
        resulting_error = np.zeros((self._input_data.shape[0], self._input_data.shape[1], image_height
                                    + (padding_top + padding_bottom), image_width +
                                    (padding_left + padding_right)), dtype=self._input_data.dtype)
        image_height = image_height + (padding_top + padding_bottom)
        image_width = image_width + (padding_top + padding_bottom)
        for output_width_idx, width_idx in enumerate(range(0, image_width - kernel_width + stride_over_width,
                                                           stride_over_width)):
            for output_height_idx, height_idx in enumerate(
                    range(0, image_height - kernel_height + stride_over_height, stride_over_height)):
                patch_gradient = (self._propagated_error[:, :, output_height_idx, output_width_idx] @
                                  np.reshape(self._weight, (-1, self._out_channels)).T).reshape((num_of_samples,
                                                                                                 self._in_channels,
                                                                                                 kernel_height,
                                                                                                 kernel_width))
                resulting_error[:, :, height_idx:(height_idx + kernel_height), width_idx:(width_idx + kernel_width)] \
                    += patch_gradient
        resulting_error = resulting_error[:, :, padding_top:(image_height -
                                                             padding_bottom), padding_left:(image_width -
                                                                                            padding_right)]
        return resulting_error
