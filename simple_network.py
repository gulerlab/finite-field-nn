# extra utils
import logging
from abc import ABC, abstractmethod

# torch
import torch
import torch.nn as nn

# this project
from utils import to_real_domain, to_finite_field_domain, finite_field_truncation


class AbstractVectorizedNet(ABC):
    def __init__(self, input_vector_size=784, hidden_layer_size=64, num_classes=10, init_fnc=nn.init.kaiming_normal_,
                 device=None, verbose=True):
        self.input_vector_size = input_vector_size
        self.hidden_layer_size = hidden_layer_size
        self.num_classes = num_classes
        self.device = device
        self.verbose = verbose
        self.init_fnc = init_fnc
        self._weight_1 = None
        self._weight_2 = None

        self.__init_weight()
        self.__to_device()

    def __init_weight(self):
        self._weight_1 = torch.empty(self.input_vector_size, self.hidden_layer_size)
        self._weight_2 = torch.empty(self.hidden_layer_size, self.num_classes)
        self.init_fnc(self._weight_1)
        self.init_fnc(self._weight_2)

        logging.info('weights are initialized using {} initialization'.format(self.init_fnc.__name__))
        if self.verbose:
            print('weights are initialized using {} initialization'.format(self.init_fnc.__name__))

    def __to_device(self):
        self._weight_1 = self._weight_1.to(self.device)
        self._weight_2 = self._weight_2.to(self.device)

        logging.info('weights are sent to {}'.format(self.device))
        if self.verbose:
            print('weights are sent to {}'.format(self.device))

    @property
    def input_vector_size(self):
        return self.input_vector_size

    @input_vector_size.setter
    def input_vector_size(self, value):
        self.input_vector_size = value

    @property
    def hidden_layer_size(self):
        return self.hidden_layer_size

    @hidden_layer_size.setter
    def hidden_layer_size(self, value):
        self.hidden_layer_size = value

    @property
    def num_classes(self):
        return self.num_classes

    @num_classes.setter
    def num_classes(self, value):
        self.num_classes = value

    @property
    def init_fnc(self):
        return self.init_fnc

    @init_fnc.setter
    def init_fnc(self, value):
        self.init_fnc = value

    @property
    def device(self):
        return self.device

    @device.setter
    def device(self, value):
        self.device = value

    @property
    def verbose(self):
        return self.verbose

    @verbose.setter
    def verbose(self, value):
        self.verbose = value

    @abstractmethod
    def criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def optimizer(self):
        pass

    @abstractmethod
    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def train(self):
        pass


class VectorizedNet(AbstractVectorizedNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

    def criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        diff = label - prediction
        diff_norm = torch.linalg.norm(diff)
        self.__save_for_backward['label'] = label
        return torch.square(diff_norm)

    def optimizer(self):
        pass

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        first_forward = torch.square(torch.matmul(torch.t(self._weight_1), input_vector))
        out = torch.matmul(torch.t(self._weight_2), first_forward)
        self.__save_for_backward = {
            'first_forward': first_forward,
            'out': out
        }
        return out

    def backward(self):
        first_forward, out, label = self.__save_for_backward['first_forward'], self.__save_for_backward['out'],\
            self.__save_for_backward['label']
        # weight_2 gradients
        weight_2_grad = -2 * torch.matmul(torch.t(label - out),
                                          torch.t(torch.kron(first_forward, torch.eye(first_forward.size(0)))))

        # weight_1 gradients

        self.__gradients = {}


    def train(self):
        pass

# class VectorizedScaledNet:
#     def __init__(self, scale_x, scale_w, input_vector_size=784, hidden_layer_size=64, num_classes=10, device=None,
#                  verbose=True) -> None:
#         self.input_vector_size = input_vector_size
#         self.hidden_layer_size = hidden_layer_size
#         self.num_classes = num_classes
#         self.device = device
#         self.verbose = verbose
#
#         self.__to_device()
#         self.__init_weight()
#
#     def __init_weight(self):
#         self.weight_1 = torch.empty(self.input_vector_size, self.hidden_layer_size)
#         self.weight_2 = torch.empty(self.hidden_layer_size, self.num_classes)
#         nn.init.kaiming_normal_(self.weight_1)
#         nn.init.kaiming_normal_(self.weight_2)
#
#     def __to_device(self):
#         self.weight_1 = self.weight_1.to(self.device)
#         self.weight_2 = self.weight_2.to(self.device)
#
#     def __init_scale(self):
#         pass
#
#     def __criterion(self):
#         pass
#
#     def __optimizer(self):
#         pass
#
#     def __scaled_forward(self):
#         pass
#
#     def __scaled_backward(self):
#         pass
#
#     def train(self):
#         pass
