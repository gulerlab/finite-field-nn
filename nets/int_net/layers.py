# this project
from nets.weight_initializations import kaiming_uniform, kaiming_uniform_conv
from nets.modules import Module
from nets.int_net.utils import int_truncation, to_int_domain, update_minmax
import numpy as np


class IntLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_bit_weight, first_layer=False,
                 quantization_bit_input=None):
        super().__init__()
        # protected
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._quantization_bit_weight = quantization_bit_weight
        self._first_layer = first_layer
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
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_bit_input(self):
        return self._quantization_bit_input

    def __init_weights(self, init_fnc=kaiming_uniform):
        self._weight = init_fnc(self._in_dim, self._out_dim)
        self._weight = to_int_domain(self._weight, self._quantization_bit_weight)

    def forward(self, input_data):
        self._input_data = input_data
        out = self._input_data @ self._weight
        if self._first_layer:
            out = int_truncation(out, self._quantization_bit_input)
        else:
            out = int_truncation(out, self._quantization_bit_weight)
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_gradient = self._input_data.T @ propagated_error
        if self._first_layer:
            scaled_gradient = int_truncation(unscaled_gradient, self._quantization_bit_input)
        else:
            scaled_gradient = int_truncation(unscaled_gradient, self._quantization_bit_weight)
        self._gradient = scaled_gradient

    def optimize(self, learning_rate):
        unscaled_gradient = self._gradient
        scaled_gradient = int_truncation(unscaled_gradient, learning_rate)
        self._weight = self._weight - scaled_gradient
        update_minmax(self._weight)

    def loss(self):
        unscaled_error = self._propagated_error @ self._weight.T
        scaled_error = int_truncation(unscaled_error, self._quantization_bit_weight)
        return scaled_error


class IntPiNetSecondOrderLinearLayer(Module):
    def __init__(self, in_dim, out_dim, quantization_bit_weight, first_layer=False, quantization_bit_input=None):
        super().__init__()
        self.__first_fc = IntLinearLayer(in_dim, out_dim, quantization_bit_weight, 
                                         first_layer=first_layer, quantization_bit_input=quantization_bit_input)
        self.__second_fc = IntLinearLayer(in_dim, out_dim, quantization_bit_weight,
                                          first_layer=first_layer, quantization_bit_input=quantization_bit_input)
        self._quantization_bit_weight = quantization_bit_weight
        self.__inner_forward = {}
        self.__inner_prop = {}

    @property
    def quantization_bit_weight(self):
        return self._quantization_bit_weight

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
        second_order = int_truncation(unscaled_second_order, self._quantization_bit_weight)
        out = second_order + first_out
        update_minmax(out)
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_second_prop = self._propagated_error * self.__inner_forward['out_1']
        second_prop = int_truncation(unscaled_second_prop, self._quantization_bit_weight)

        unscaled_first_prop_second_order = self._propagated_error * self.__inner_forward['out_2']
        first_prop_second_order = int_truncation(unscaled_first_prop_second_order,
                                                          self._quantization_bit_weight)
        first_prop = first_prop_second_order + self._propagated_error
        update_minmax(first_prop)
        self.__inner_prop['prop_1'] = first_prop
        self.__inner_prop['prop_2'] = second_prop

        self.__first_fc.backprop(first_prop)
        self.__second_fc.backprop(second_prop)

    def optimize(self, learning_rate):
        self.__first_fc.optimize(learning_rate)
        self.__second_fc.optimize(learning_rate)

    def loss(self):
        total_loss = self.__first_fc.loss() + self.__second_fc.loss()
        update_minmax(total_loss) 
        return total_loss


class IntConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, quantization_bit_weight,
                 stride=(1, 1), padding=(0, 0, 0, 0), first_layer=False, quantization_bit_input=None):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._quantization_bit_weight = quantization_bit_weight
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
    def first_layer(self):
        return self._first_layer

    @property
    def quantization_bit_input(self):
        return self._quantization_bit_input

    def __init_weights(self, init_fnc=kaiming_uniform_conv):
        self._weight = init_fnc(self._in_channels, self._out_channels, self._kernel_size)
        self._weight = to_int_domain(self._weight, self._quantization_bit_weight)

    def __generate_patches(self):
        self.__patches = []
        num_of_samples, _, image_height, image_width = self._input_data.shape
        kernel_height, kernel_width = self._kernel_size
        stride_over_height, stride_over_width = self._stride
        padding_top, padding_bottom, padding_left, padding_right = self._padding
        curr_input_data = np.zeros((self._input_data.shape[0], self._input_data.shape[1], image_height
                                             + (padding_top + padding_bottom), image_width +
                                             (padding_left + padding_right)), dtype=np.int64)
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
        output_data = np.zeros((num_of_samples, self._out_channels, output_height, output_width), dtype=np.int64)
        for patch in self.__patches:
            output_height_idx, output_width_idx, patch_data = patch
            unscaled_out = (np.reshape(patch_data, (num_of_samples, -1)) @
                            np.reshape(self._weight, (-1, self._out_channels)))
            output_data[:, :, output_height_idx, output_width_idx] = unscaled_out
        if self._first_layer:
            output_data = int_truncation(output_data, self._quantization_bit_input)
        else:
            output_data = int_truncation(output_data, self._quantization_bit_weight)
        return output_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        self._gradient = np.zeros(self._weight.shape, dtype=np.int64)
        for patch in self.__patches:
            output_height_idx, output_width_idx, patch_data = patch
            unscaled_gradient = ((
                                         np.reshape(patch_data, (patch_data.shape[0], -1)).T @
                                         self._propagated_error[:, :, output_height_idx, output_width_idx]
                                 ).reshape(self._weight.shape))

            self._gradient = self._gradient + unscaled_gradient
            update_minmax(self._gradient)
        if self._first_layer:
            self._gradient = int_truncation(self._gradient, self._quantization_bit_input)
        else:
            self._gradient = int_truncation(self._gradient, self._quantization_bit_weight)

    def optimize(self, learning_rate):
        unscaled_gradient = self._gradient
        scaled_gradient = int_truncation(unscaled_gradient, learning_rate)
        self._weight = self._weight - scaled_gradient
        update_minmax(self._weight)

    def loss(self):
        num_of_samples, _, image_height, image_width = self._input_data.shape
        kernel_height, kernel_width = self._kernel_size
        stride_over_height, stride_over_width = self._stride
        padding_top, padding_bottom, padding_left, padding_right = self._padding
        resulting_error = np.zeros((self._input_data.shape[0], self._input_data.shape[1], image_height
                                             + (padding_top + padding_bottom), image_width +
                                             (padding_left + padding_right)), np.int64)
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
        resulting_error = int_truncation(resulting_error, self._quantization_bit_weight)
        return resulting_error


class IntPiNetSecondOrderConvLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, quantization_bit_weight,
                 stride=(1, 1), padding=(0, 0, 0, 0), first_layer=False, quantization_bit_input=None):
        super().__init__()
        self.__first_conv = IntConvLayer(in_channels, out_channels, kernel_size, quantization_bit_weight,
                                         stride=stride, padding=padding, first_layer=first_layer,
                                         quantization_bit_input=quantization_bit_input)
        self.__second_conv = IntConvLayer(in_channels, out_channels, kernel_size, quantization_bit_weight, stride=stride,
                                          padding=padding, first_layer=first_layer,
                                          quantization_bit_input=quantization_bit_input)
        self._quantization_bit_weight = quantization_bit_weight
        self.__inner_forward = {}
        self.__inner_prop = {}

    @property
    def quantization_bit_weight(self):
        return self._quantization_bit_weight
    
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
        second_order = int_truncation(unscaled_second_order, self._quantization_bit_weight)
        out = second_order + first_out
        update_minmax(out)
        return out

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        unscaled_second_prop = self._propagated_error * self.__inner_forward['out_1']
        second_prop = int_truncation(unscaled_second_prop, self._quantization_bit_weight)

        unscaled_first_prop_second_order = self._propagated_error * self.__inner_forward['out_2']
        first_prop_second_order = int_truncation(unscaled_first_prop_second_order,
                                                          self._quantization_bit_weight)
        first_prop = first_prop_second_order + self._propagated_error
        update_minmax(first_prop)
        self.__inner_prop['prop_1'] = first_prop
        self.__inner_prop['prop_2'] = second_prop

        self.__first_conv.backprop(first_prop)
        self.__second_conv.backprop(second_prop)

    def optimize(self, learning_rate):
        self.__first_conv.optimize(learning_rate)
        self.__second_conv.optimize(learning_rate)

    def loss(self):
        total_loss = self.__first_conv.loss() + self.__second_conv.loss()
        update_minmax(total_loss) 
        return total_loss
