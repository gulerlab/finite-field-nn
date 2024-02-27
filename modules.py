from abc import ABC, abstractmethod
import numpy as np
import copy
from collections import OrderedDict
from re import finditer
import os
import pickle

class Module(ABC):
    def __init__(self):
        super().__init__()
        # protected
        self._input_data = None
        self._gradient = None
        self._weight = None
        self._propagated_error = None

    @property
    def input_data(self):
        return self._input_data
    
    @property
    def weight(self):
        return copy.deepcopy(self._weight)
    
    @weight.setter
    def weight(self, value):
        self._weight = value

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


class Network(Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    @property
    def model(self):
        return self._model

    def forward(self, input_data):
        self._input_data = input_data
        curr_data = self._input_data
        for layer in self._model:
            curr_data = layer.forward(curr_data)
        return curr_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        curr_error = self._propagated_error
        for layer in reversed(self._model):
            layer.backprop(curr_error)
            curr_error = layer.loss()

    def optimize(self, learning_rate):
        for layer in self._model:
            if isinstance(layer, Module):
                layer.optimize(learning_rate)

    def loss(self):
        pass

    def return_all_weights(self):
        iterator = 0
        dict_elem_arr = []
        for layer in self._model:
            if isinstance(layer, Module):
                curr_weight = layer.weight
                class_name = type(layer).__name__
                class_name = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', class_name)
                class_name = [m.group(0) for m in class_name]
                class_name = [m.lower() for m in class_name]
                class_name.append(str(iterator))
                if isinstance(curr_weight, tuple):
                    first, second = curr_weight
                    first_name = '.'.join(class_name + ['0'])
                    dict_elem_arr.append((first_name, first))
                    second_name = '.'.join(class_name + ['1'])
                    dict_elem_arr.append((second_name, second))
                else:
                    class_name = '.'.join(class_name)
                    dict_elem_arr.append((class_name, curr_weight))
                iterator = iterator + 1
        return OrderedDict(dict_elem_arr)
    
    def save_all_weights(self, save_path):
        weights = self.return_all_weights()
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])

        with open(save_path, 'wb') as fp:
            pickle.dump(weights, fp, pickle.HIGHEST_PROTOCOL)

    def load_all_weights(self, load_path):
        with open(load_path, 'rb') as fp:
            weights = pickle.load(fp)
        iterator = 0
        for layer in self._model:
            if isinstance(layer, Module):
                class_name = type(layer).__name__
                class_name = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', class_name)
                class_name = [m.group(0) for m in class_name]
                class_name = [m.lower() for m in class_name]
                class_name.append(str(iterator))
                if 'pi' in class_name:
                    first_name = '.'.join(class_name + ['0'])
                    second_name = '.'.join(class_name + ['1'])
                    layer.weight = (weights[first_name], weights[second_name])
                else:
                    class_name = '.'.join(class_name)
                    layer.weight = weights[class_name]
                iterator = iterator + 1
        

class ActivationModule(ABC):
    def __init__(self):
        super().__init__()
        # protected
        self._input_data = None
        self._gradient = None
        self._propagated_error = None

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backprop(self, propagated_error):
        pass

    @abstractmethod
    def loss(self):
        pass


class Flatten(ActivationModule):
    def __init__(self):
        super().__init__()
        self._input_data_shape = None

    def forward(self, input_data):
        self._input_data_shape = input_data.shape
        return np.reshape(input_data, (self._input_data_shape[0], -1))

    def backprop(self, propagated_error):
        self._gradient = np.reshape(propagated_error, self._input_data_shape)

    def loss(self):
        return self._gradient
