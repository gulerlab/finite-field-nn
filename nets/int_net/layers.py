# this project
from weight_initializations import kaiming_uniform, kaiming_uniform_conv
from modules import Module
from utils import to_int_domain, int_truncation

import numpy as np


class IntegerLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_weight, first_layer=False, quantization_input=None):
        super().__init__()
        # protected
        self._in_dim = in_dim
        self._out_dim = out_dim

        self._quantization_weight = quantization_weight
        self._first_layer = first_layer
        self._quantization_input = quantization_input

        self.__init_weights()

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def quantization_weight(self):
        return self._quantization_weight

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_input(self):
        return self._quantization_input

    def __init_weights(self, init_fnc=kaiming_uniform):
        self._weight = init_fnc(self._in_dim, self._out_dim)
        self._weight = to_int_domain(self._weight, self._quantization_weight)

    def forward(self, input_data):
        self._input_data = input_data
        unscaled_result = self._input_data @ self._weight
        if self._first_layer:
            scaled_result = int_truncation(unscaled_result, self._quantization_input)
        else:
            scaled_result = int_truncation(unscaled_result, self._quantization_weight)
        return scaled_result

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_gradient = self._input_data.T @ propagated_error
        if self._first_layer:
            scaled_gradient = int_truncation(unscaled_gradient, self._quantization_input)
        else:
            scaled_gradient = int_truncation(unscaled_gradient, self._quantization_weight)
        self._gradient = scaled_gradient

    def optimize(self, learning_rate):
        self._weight = self._weight - int_truncation(self._gradient, learning_rate)

    def loss(self):
        return int_truncation(self._propagated_error @ self._weight.T, self._quantization_weight)


class IntegerPiNetSecondOrderLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_weight, first_layer=False, quantization_input=None):
        super().__init__()
        self._quantization_weight = quantization_weight
        self._first_layer = first_layer
        self._quantization_input = quantization_input
        self.__first_fc = IntegerLinearLayer(in_dim, out_dim, quantization_weight, first_layer=first_layer,
                                             quantization_input=quantization_input)
        self.__second_fc = IntegerLinearLayer(in_dim, out_dim,  quantization_weight, first_layer=first_layer,
                                              quantization_input=quantization_input)
        self.__inner_forward = {}
        self.__inner_prop = {}

    @property
    def quantization_weight(self):
        return self._quantization_weight

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_input(self):
        return self._quantization_input

    def forward(self, input_data):
        self._input_data = input_data
        first_out = self.__first_fc.forward(input_data)
        second_out = self.__second_fc.forward(input_data)
        self.__inner_forward['out_1'] = first_out
        self.__inner_forward['out_2'] = second_out
        unscaled_second_order = (first_out * second_out)
        scaled_second_order = int_truncation(unscaled_second_order, self._quantization_weight)
        return scaled_second_order + first_out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_second_prop = self._propagated_error * self.__inner_forward['out_1']
        second_prop = int_truncation(unscaled_second_prop, self._quantization_weight)
        unscaled_first_prop = self._propagated_error * self.__inner_forward['out_2']
        first_prop = int_truncation(unscaled_first_prop, self._quantization_weight)
        first_prop = first_prop + self._propagated_error
        self.__inner_prop['prop_1'] = first_prop
        self.__inner_prop['prop_2'] = second_prop

        self.__first_fc.backprop(first_prop)
        self.__second_fc.backprop(second_prop)

    def optimize(self, learning_rate):
        self.__first_fc.optimize(learning_rate)
        self.__second_fc.optimize(learning_rate)

    def loss(self):
        return self.__first_fc.loss() + self.__second_fc.loss()


class IntegerConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, quantization_weight, stride=(1, 1), padding=(0, 0, 0, 0),
                 first_layer=False, quantization_input=None):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._quantization_weight = quantization_weight
        self._first_layer = first_layer
        self._quantization_input = quantization_input
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

    @property
    def quantization_weight(self):
        return self._quantization_weight

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_input(self):
        return self._quantization_input

    def __init_weights(self, init_fnc=kaiming_uniform_conv):
        self._weight = init_fnc(self._in_channels, self._out_channels, self._kernel_size)
        self._weight = to_int_domain(self._weight, self._quantization_weight)

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
        if self._first_layer:
            output_data = int_truncation(output_data, self._quantization_input)
        else:
            output_data = int_truncation(output_data, self._quantization_weight)
        return output_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = np.zeros(self._weight.shape, dtype=self._weight.dtype)
        for patch in self.__patches:
            output_height_idx, output_width_idx, patch_data = patch
            self._gradient += (np.reshape(patch_data, (patch_data.shape[0], -1)).T
                               @ self._propagated_error[:, :, output_height_idx, output_width_idx])\
                .reshape(self._weight.shape)

        if self._first_layer:
            self._gradient = int_truncation(self._gradient, self._quantization_input)
        else:
            self._gradient = int_truncation(self._gradient, self._quantization_weight)

    def optimize(self, learning_rate):
        self._weight = self._weight - int_truncation(self._gradient, learning_rate)

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
        resulting_error = int_truncation(resulting_error, self._quantization_weight)
        return resulting_error


class IntegerPiNetSecondOrderConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, quantization_weight, first_layer=False,
                 quantization_input=None):
        super().__init__()
        self._quantization_weight = quantization_weight
        self._first_layer = first_layer
        self._quantization_input = quantization_input
        self.__first_conv = IntegerConvLayer(in_channels, out_channels, kernel_size, quantization_weight,
                                             first_layer=first_layer, quantization_input=quantization_input)
        self.__second_conv = IntegerConvLayer(in_channels, out_channels, kernel_size, quantization_weight,
                                              first_layer=first_layer, quantization_input=quantization_input)
        self.__inner_forward = {}
        self.__inner_prop = {}

    @property
    def quantization_weight(self):
        return self._quantization_weight

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_input(self):
        return self._quantization_input

    def forward(self, input_data):
        self._input_data = input_data
        first_out = self.__first_conv.forward(input_data)
        second_out = self.__second_conv.forward(input_data)
        self.__inner_forward['out_1'] = first_out
        self.__inner_forward['out_2'] = second_out
        unscaled_second_order = (first_out * second_out)
        scaled_second_order = int_truncation(unscaled_second_order, self._quantization_weight)
        return scaled_second_order + first_out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_second_prop = self._propagated_error * self.__inner_forward['out_1']
        second_prop = int_truncation(unscaled_second_prop, self._quantization_weight)
        unscaled_first_prop = self._propagated_error * self.__inner_forward['out_2']
        first_prop = int_truncation(unscaled_first_prop, self._quantization_weight)
        first_prop = first_prop + self._propagated_error
        self.__inner_prop['prop_1'] = first_prop
        self.__inner_prop['prop_2'] = second_prop

        self.__first_conv.backprop(first_prop)
        self.__second_conv.backprop(second_prop)

    def optimize(self, learning_rate):
        self.__first_conv.optimize(learning_rate)
        self.__second_conv.optimize(learning_rate)

    def loss(self):
        return self.__first_conv.loss() + self.__second_conv.loss()
