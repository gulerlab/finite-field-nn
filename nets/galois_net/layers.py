# this project
from nets.weight_initializations import kaiming_uniform, kaiming_uniform_conv
from nets.modules import Module
from nets.galois_net.utils import finite_field_truncation, to_finite_field_domain
import numpy as np


class GaloisFieldLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_bit_weight, prime, field, first_layer=False,
                 quantization_bit_input=None):
        super().__init__()
        # protected
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._quantization_bit_weight = quantization_bit_weight
        self._prime = prime
        self._first_layer = first_layer
        self._field = field
        if self._first_layer:
            assert quantization_bit_input is not None, 'if the layer is the first layer then the input' \
                                                       ' quantization bit required'
            self._quantization_bit_input = quantization_bit_input
        self.__init_weights()

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def quantization_bit_weight(self):
        return self._quantization_bit_weight

    @property
    def prime(self):
        return self._prime

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_bit_input(self):
        return self._quantization_bit_input

    @property
    def field(self):
        return self._field

    def __init_weights(self, init_fnc=kaiming_uniform):
        self._weight = init_fnc(self._in_dim, self._out_dim)
        self._weight = to_finite_field_domain(self._weight, self._quantization_bit_weight, self._prime, self._field)

    def forward(self, input_data):
        self._input_data = input_data
        out = self._input_data @ self._weight
        if self._first_layer:
            out = finite_field_truncation(out, self._quantization_bit_input, self._prime, self._field)
        else:
            out = finite_field_truncation(out, self._quantization_bit_weight, self._prime, self._field)
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_gradient = self._input_data.T @ propagated_error
        if self._first_layer:
            scaled_gradient = finite_field_truncation(unscaled_gradient, self._quantization_bit_input, self._prime, self._field)
        else:
            scaled_gradient = finite_field_truncation(unscaled_gradient, self._quantization_bit_weight, self._prime, self._field)
        self._gradient = scaled_gradient

    def optimize(self, learning_rate):
        unscaled_gradient = self._gradient
        scaled_gradient = finite_field_truncation(unscaled_gradient, learning_rate, self._prime, self._field)
        self._weight = self._weight - scaled_gradient

    def loss(self):
        unscaled_error = self._propagated_error @ self._weight.T
        scaled_error = finite_field_truncation(unscaled_error, self._quantization_bit_weight, self._prime, self._field)
        return scaled_error


class GaloisFieldPiNetSecondOrderLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_bit_weight, prime, field,
                 first_layer=False, quantization_bit_input=None):
        super().__init__()
        self.__first_fc = GaloisFieldLinearLayer(in_dim, out_dim, quantization_bit_weight, prime, field,
                                                 first_layer=first_layer, quantization_bit_input=quantization_bit_input)
        self.__second_fc = GaloisFieldLinearLayer(in_dim, out_dim, quantization_bit_weight, prime, field,
                                                  first_layer=first_layer,
                                                  quantization_bit_input=quantization_bit_input)
        self._prime = prime
        self._quantization_bit_weight = quantization_bit_weight
        self._field = field
        self.__inner_forward = {}
        self.__inner_prop = {}

    @property
    def quantization_bit_weight(self):
        return self._quantization_bit_weight

    @property
    def prime(self):
        return self._prime

    @property
    def field(self):
        return self._field

    @property
    def weight(self):
        return self.__first_fc.weight, self.__second_fc.weight
    
    @weight.setter
    def weight(self, value):
        first, second = value
        self.__first_fc.weight = first
        self.__second_fc.weight = second
    
    def forward(self, input_data):
        self._input_data = input_data
        first_out = self.__first_fc.forward(input_data)
        second_out = self.__second_fc.forward(input_data)
        self.__inner_forward['out_1'] = first_out
        self.__inner_forward['out_2'] = second_out
        unscaled_second_order = first_out * second_out
        second_order = finite_field_truncation(unscaled_second_order, self._quantization_bit_weight, self._prime, self._field)
        out = second_order + first_out
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_second_prop = self._propagated_error * self.__inner_forward['out_1']
        second_prop = finite_field_truncation(unscaled_second_prop, self._quantization_bit_weight, self._prime, self._field)

        unscaled_first_prop_second_order = self._propagated_error * self.__inner_forward['out_2']
        first_prop_second_order = finite_field_truncation(unscaled_first_prop_second_order,
                                                          self._quantization_bit_weight, self._prime, self._field)
        first_prop = first_prop_second_order + self._propagated_error
        self.__inner_prop['prop_1'] = first_prop
        self.__inner_prop['prop_2'] = second_prop

        self.__first_fc.backprop(first_prop)
        self.__second_fc.backprop(second_prop)

    def optimize(self, learning_rate):
        self.__first_fc.optimize(learning_rate)
        self.__second_fc.optimize(learning_rate)

    def loss(self):
        return self.__first_fc.loss() + self.__second_fc.loss()


class GaloisFieldConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, quantization_bit_weight, prime, field,
                 stride=(1, 1), padding=(0, 0, 0, 0), first_layer=False, quantization_bit_input=None):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._quantization_bit_weight = quantization_bit_weight
        self._prime = prime
        self._field = field
        self._first_layer = first_layer
        if self._first_layer:
            assert quantization_bit_input is not None, 'if the layer is the first layer then the input' \
                                                       ' quantization bit required'
            self._quantization_bit_input = quantization_bit_input
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
    def quantization_bit_weight(self):
        return self._quantization_bit_weight

    @property
    def prime(self):
        return self._prime

    def field(self):
        return self._field

    @property
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_bit_input(self):
        return self._quantization_bit_input

    def __init_weights(self, init_fnc=kaiming_uniform_conv):
        self._weight = init_fnc(self._in_channels, self._out_channels, self._kernel_size)
        self._weight = to_finite_field_domain(self._weight, self._quantization_bit_weight, self._prime, self._field)

    def __generate_patches(self):
        self.__patches = []
        num_of_samples, _, image_height, image_width = self._input_data.shape
        kernel_height, kernel_width = self._kernel_size
        stride_over_height, stride_over_width = self._stride
        padding_top, padding_bottom, padding_left, padding_right = self._padding
        curr_input_data = self._field.Zeros((self._input_data.shape[0], self._input_data.shape[1], image_height
                                             + (padding_top + padding_bottom), image_width +
                                             (padding_left + padding_right)))
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
        output_height, output_width = (int((image_height - kernel_height + stride_over_height) / stride_over_height),
                                       int((image_width - kernel_width + stride_over_width) / stride_over_width))
        output_data = self._field.Zeros((num_of_samples, self._out_channels, output_height, output_width))
        for patch in self.__patches:
            output_height_idx, output_width_idx, patch_data = patch
            unscaled_out = (np.reshape(patch_data, (num_of_samples, -1)) @
                            np.reshape(self._weight, (-1, self._out_channels)))
            output_data[:, :, output_height_idx, output_width_idx] = unscaled_out
        if self._first_layer:
            output_data = finite_field_truncation(output_data, self._quantization_bit_input, self._prime, self._field)
        else:
            output_data = finite_field_truncation(output_data, self._quantization_bit_weight, self._prime, self._field)
        return output_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = self._field.Zeros(self._weight.shape)
        for patch in self.__patches:
            output_height_idx, output_width_idx, patch_data = patch
            unscaled_gradient = ((
                                         np.reshape(patch_data, (patch_data.shape[0], -1)).T @
                                         self._propagated_error[:, :, output_height_idx, output_width_idx]
                                 ).reshape(self._weight.shape))

            self._gradient = self._gradient + unscaled_gradient
        if self._first_layer:
            self._gradient = finite_field_truncation(self._gradient, self._quantization_bit_input, self._prime, self._field)
        else:
            self._gradient = finite_field_truncation(self._gradient, self._quantization_bit_weight, self._prime, self._field)

    def optimize(self, learning_rate):
        unscaled_gradient = self._gradient
        scaled_gradient = finite_field_truncation(unscaled_gradient, learning_rate, self._prime, self._field)
        self._weight = self._weight - scaled_gradient

    def loss(self):
        num_of_samples, _, image_height, image_width = self._input_data.shape
        kernel_height, kernel_width = self._kernel_size
        stride_over_height, stride_over_width = self._stride
        padding_top, padding_bottom, padding_left, padding_right = self._padding
        resulting_error = self._field.Zeros((self._input_data.shape[0], self._input_data.shape[1], image_height
                                             + (padding_top + padding_bottom), image_width +
                                             (padding_left + padding_right)))
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
        resulting_error = finite_field_truncation(resulting_error, self._quantization_bit_weight, self._prime, self._field)
        return resulting_error


class GaloisFieldPiNetSecondOrderConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, quantization_bit_weight, prime, field,
                 stride=(1, 1), padding=(0, 0, 0, 0), first_layer=False, quantization_bit_input=None):
        super().__init__()
        self.__first_conv = GaloisFieldConvLayer(in_channels, out_channels, kernel_size, quantization_bit_weight, prime,
                                                 field, stride=stride, padding=padding, first_layer=first_layer,
                                                 quantization_bit_input=quantization_bit_input)
        self.__second_conv = GaloisFieldConvLayer(in_channels, out_channels, kernel_size, quantization_bit_weight,
                                                  prime, field, stride=stride, padding=padding, first_layer=first_layer,
                                                  quantization_bit_input=quantization_bit_input)
        self._prime = prime
        self._field = field
        self._quantization_bit_weight = quantization_bit_weight
        self.__inner_forward = {}
        self.__inner_prop = {}

    @property
    def quantization_bit_weight(self):
        return self._quantization_bit_weight

    @property
    def prime(self):
        return self._prime

    @property
    def field(self):
        return self._field
    
    @property
    def weight(self):
        return self.__first_conv.weight, self.__second_conv.weight
    
    @weight.setter
    def weight(self, value):
        first, second = value
        self.__first_conv.weight = first
        self.__second_conv.weight = second

    def forward(self, input_data):
        self._input_data = input_data
        first_out = self.__first_conv.forward(input_data)
        second_out = self.__second_conv.forward(input_data)
        self.__inner_forward['out_1'] = first_out
        self.__inner_forward['out_2'] = second_out
        unscaled_second_order = first_out * second_out
        second_order = finite_field_truncation(unscaled_second_order, self._quantization_bit_weight, self._prime, self._field)
        out = second_order + first_out
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_second_prop = self._propagated_error * self.__inner_forward['out_1']
        second_prop = finite_field_truncation(unscaled_second_prop, self._quantization_bit_weight, self._prime, self._field)

        unscaled_first_prop_second_order = self._propagated_error * self.__inner_forward['out_2']
        first_prop_second_order = finite_field_truncation(unscaled_first_prop_second_order,
                                                          self._quantization_bit_weight, self._prime, self._field)
        first_prop_second_order = self._field(first_prop_second_order)
        first_prop = first_prop_second_order + self._propagated_error
        self.__inner_prop['prop_1'] = first_prop
        self.__inner_prop['prop_2'] = second_prop

        self.__first_conv.backprop(first_prop)
        self.__second_conv.backprop(second_prop)

    def optimize(self, learning_rate):
        self.__first_conv.optimize(learning_rate)
        self.__second_conv.optimize(learning_rate)

    def loss(self):
        return self.__first_conv.loss() + self.__second_conv.loss()
