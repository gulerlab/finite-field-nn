# extra utils
import logging
from abc import ABC, abstractmethod
import math

# torch
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# this project
from utils import to_real_domain, to_finite_field_domain, finite_field_truncation, ToFiniteFieldDomain


class AbstractVectorizedNet(ABC):
    def __init__(self, input_vector_size=784, hidden_layer_size=64, num_classes=10,
                 device=None, verbose=True):
        self.__input_vector_size = input_vector_size
        self.__hidden_layer_size = hidden_layer_size
        self.__num_classes = num_classes
        if device is None:
            self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.__device = device
        self.__verbose = verbose
        self._weight_1 = None
        self._weight_2 = None

        self.__init_weight()
        self.__to_device()

    def __init_weight(self):
        range_low, range_high = -1 / math.sqrt(self.__input_vector_size), 1 / math.sqrt(self.__input_vector_size)
        self._weight_1 = range_low + torch.rand((self.__input_vector_size, self.__hidden_layer_size)) * (range_high -
                                                                                                         range_low)
        range_low, range_high = -1 / math.sqrt(self.__hidden_layer_size), 1 / math.sqrt(self.__hidden_layer_size)
        self._weight_2 = range_low + torch.rand((self.__hidden_layer_size, self.__num_classes)) * (range_high -
                                                                                                   range_low)
        logging.info('weights are initialized')
        if self.__verbose:
            print('weights are initialized')

    def __to_device(self):
        self._weight_1 = self._weight_1.to(self.__device)
        self._weight_2 = self._weight_2.to(self.__device)

        logging.info('weights are sent to {}'.format(self.__device))
        if self.__verbose:
            print('weights are sent to {}'.format(self.__device))

    @property
    def input_vector_size(self):
        return self.__input_vector_size

    @input_vector_size.setter
    def input_vector_size(self, value):
        self.__input_vector_size = value

    @property
    def hidden_layer_size(self):
        return self.__hidden_layer_size

    @hidden_layer_size.setter
    def hidden_layer_size(self, value):
        self.__hidden_layer_size = value

    @property
    def num_classes(self):
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, value):
        self.__num_classes = value

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value

    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, value):
        self.__verbose = value

    @abstractmethod
    def _criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _optimizer(self, learning_rate):
        pass

    @abstractmethod
    def _forward(self, input_vector: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        pass

    @abstractmethod
    def _backward(self):
        pass

    @abstractmethod
    def train(self, data_path: str, num_of_epochs: int, learning_rate):
        pass


class VectorizedNet(AbstractVectorizedNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

    def _criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        diff = label - prediction
        diff_norm = torch.linalg.norm(diff)
        self.__save_for_backward['label'] = label
        return torch.square(diff_norm)

    def _optimizer(self, learning_rate: float):
        self._weight_2 = self._weight_2 - learning_rate * self.__gradients['weight_2_grad']
        self._weight_1 = self._weight_1 - learning_rate * self.__gradients['weight_1_grad']

    def _forward(self, input_vector: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        first_forward = torch.square(torch.matmul(torch.t(self._weight_1), input_vector))
        out = torch.matmul(torch.t(self._weight_2), first_forward)
        if mode == 'train':
            self.__save_for_backward = {
                'input_vector': input_vector,
                'first_forward': first_forward,
                'out': out
            }

        return out

    def _backward(self):
        first_forward, out, label, input_vector = self.__save_for_backward['first_forward'], \
            self.__save_for_backward['out'], self.__save_for_backward['label'], self.__save_for_backward['input_vector']

        # weight_2 gradients
        # weight_2_grad = -2 * torch.matmul(torch.t(label - out),
        #                                   torch.t(torch.kron(first_forward, torch.eye(self._weight_2.size(1)))))
        weight_2_grad = -2 * torch.matmul(first_forward, torch.t(label - out))

        # weight_1 gradients
        # first_chain = torch.t(torch.kron(input_vector, torch.eye(self._weight_1.size(1))))
        second_chain = 2 * torch.diag(torch.matmul(torch.t(self._weight_1), input_vector).reshape(-1))
        third_chain = torch.t(self._weight_2)
        fourth_chain = -2 * torch.t(label - out)
        # weight_1_grad = torch.matmul(second_chain, first_chain)
        weight_1_grad = second_chain
        weight_1_grad = torch.matmul(third_chain, weight_1_grad)
        weight_1_grad = torch.matmul(fourth_chain, weight_1_grad)
        weight_1_grad = torch.matmul(input_vector, weight_1_grad)

        # self.__gradients = {
        #     'weight_2_grad': weight_2_grad.T.reshape(self._weight_2.size(1), self._weight_2.size(0)).T,
        #     'weight_1_grad': weight_1_grad.T.reshape(self._weight_1.size(1), self._weight_1.size(0)).T
        # }

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    def train(self, data_path: str, num_of_epochs: int, learning_rate: float):
        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))
        # load data
        train_dataset = FashionMNIST(data_path, train=True, transform=transform, target_transform=target_transform,
                                     download=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        test_dataset = FashionMNIST(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        running_loss = []
        running_acc = []
        curr_loss = torch.zeros(1).to(self.device)
        curr_acc = 0
        for epoch in range(num_of_epochs):
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                data, label = data.squeeze().T.reshape(-1, 1), label.reshape(-1, 1)

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss

                if idx == 0 or (idx + 1) % 10000 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss.item())
                    else:
                        running_loss.append((curr_loss / 10000).item())
                    test_idx = 1
                    for test_data, test_label in test_loader:
                        test_data, test_label = test_data.to(self.device), test_label.to(self.device)
                        test_data = test_data.squeeze().T.reshape(-1, 1)
                        test_out = self._forward(test_data, mode='eval')
                        pred_label = torch.argmax(test_out)
                        if pred_label == test_label:
                            curr_acc = curr_acc + 1
                        test_idx = test_idx + 1
                    running_acc.append(curr_acc / (test_idx + 1))
                    print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0


class SimpleNetwork(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.hidden_layer = nn.Linear(784, 64, bias=False)
        self.output_layer = nn.Linear(64, num_class, bias=False)

    def forward(self, data):
        data = data.squeeze().T.reshape(-1)
        data = self.hidden_layer(data)
        data = torch.square(data)
        data = self.output_layer(data)
        return data


class ScaledVectorizedNet(AbstractVectorizedNet):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, prime, **kwargs):
        super().__init__(**kwargs)
        self.__scale_input_parameter = scale_input_parameter
        self.__scale_weight_parameter = scale_weight_parameter
        self.__scale_learning_rate_parameter = scale_learning_rate_parameter
        self.__prime = prime

        self.__save_for_backward = {}
        self.__gradients = {}

    def __scale_init_weight(self):
        self._weight_1 = to_finite_field_domain(self._weight_1, self.__scale_weight_parameter, self.__prime)
        self._weight_2 = to_finite_field_domain(self._weight_2, self.__scale_weight_parameter, self.__prime)

    @property
    def scale_input_parameter(self):
        return self.__scale_input_parameter

    @scale_input_parameter.setter
    def scale_input_parameter(self, value):
        self.__scale_input_parameter = value

    @property
    def scale_weight_parameter(self):
        return self.__scale_weight_parameter

    @scale_weight_parameter.setter
    def scale_weight_parameter(self, value):
        self.__scale_weight_parameter = value

    @property
    def prime(self):
        return self.__prime

    @prime.setter
    def prime(self, value):
        self.__prime = value

    @property
    def scale_learning_rate_parameter(self):
        return self.__scale_learning_rate_parameter

    @scale_learning_rate_parameter.setter
    def scale_learning_rate_parameter(self, value):
        self.__scale_learning_rate_parameter = value

    def _criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        diff = label - prediction
        diff_norm = torch.linalg.norm(diff)
        self.__save_for_backward['label'] = label
        return torch.square(diff_norm)

    def _optimizer(self, learning_rate: int):
        weight_1_grad_out, weight_1_grad_label = self.__gradients['weight_1_grad_out'], \
            self.__gradients['weight_1_grad_label']
        weight_2_grad_out, weight_2_grad_label = self.__gradients['weight_2_grad_out'], \
            self.__gradients['weight_2_grad_label']

        out_scale = 4 * (self.__scale_input_parameter +
                         self.__scale_weight_parameter) + self.__scale_learning_rate_parameter
        label_scale = 2 * (self.__scale_input_parameter +
                           self.__scale_weight_parameter) + self.__scale_learning_rate_parameter
        weight_1_grad_out = finite_field_truncation(learning_rate * weight_1_grad_out, out_scale)
        weight_1_grad_label = finite_field_truncation(learning_rate * weight_1_grad_label, label_scale)
        weight_2_grad_out = finite_field_truncation(learning_rate * weight_2_grad_out, out_scale)
        weight_2_grad_label = finite_field_truncation(learning_rate * weight_2_grad_label, label_scale)

        weight_1_grad = weight_1_grad_label + weight_1_grad_out
        weight_2_grad = weight_2_grad_label + weight_2_grad_out

        self._weight_1 = self._weight_1 - weight_1_grad
        self._weight_2 = self._weight_2 - weight_2_grad

    def _forward(self, input_vector: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        first_forward = torch.square(torch.matmul(torch.t(self._weight_1), input_vector))
        out = torch.matmul(torch.t(self._weight_2), first_forward)

        if mode == 'train':
            self.__save_for_backward['input_vector'] = input_vector
            self.__save_for_backward['first_forward'] = first_forward
            self.__save_for_backward['out'] = out

        return finite_field_truncation(out, 2 * (self.__scale_weight_parameter + self.__scale_input_parameter))

    def _backward(self):
        first_forward, out, label, input_vector = self.__save_for_backward['first_forward'], \
            self.__save_for_backward['out'], self.__save_for_backward['label'], self.__save_for_backward['input_vector']

        weight_2_grad_out = 2 * torch.matmul(first_forward, torch.t(out))
        weight_2_grad_label = -2 * torch.matmul(first_forward, torch.t(label))
        self.__gradients['weight_2_grad_out'] = weight_2_grad_out
        self.__gradients['weight_2_grad_label'] = weight_2_grad_label

        second_chain = 2 * torch.diag(torch.matmul(torch.t(self._weight_1), input_vector).reshape(-1))
        third_chain = torch.t(self._weight_2)
        fourth_chain_out = 2 * torch.t(out)
        fourth_chain_label = -2 * torch.t(label)

        weight_1_grad_out = second_chain
        weight_1_grad_out = torch.matmul(third_chain, weight_1_grad_out)
        weight_1_grad_out = torch.matmul(fourth_chain_out, weight_1_grad_out)
        weight_1_grad_out = torch.matmul(input_vector, weight_1_grad_out)

        weight_1_grad_label = second_chain
        weight_1_grad_label = torch.matmul(third_chain, weight_1_grad_label)
        weight_1_grad_label = torch.matmul(fourth_chain_label, weight_1_grad_label)
        weight_1_grad_label = torch.matmul(input_vector, weight_1_grad_label)

        self.__gradients['weight_1_grad_out'] = weight_1_grad_out
        self.__gradients['weight_1_grad_label'] = weight_1_grad_label

    def train(self, data_path: str, num_of_epochs: int, learning_rate: float):
        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            ToFiniteFieldDomain(self.__scale_input_parameter, self.__prime)
        ])

        # target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
        #                                      .scatter_(0, torch.tensor(y), 1))
        # # load data
        # train_dataset = FashionMNIST(data_path, train=True, transform=transform, target_transform=target_transform,
        #                              download=True)
        # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        #
        # test_dataset = FashionMNIST(data_path, train=False, transform=transform, download=True)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        #
        # running_loss = []
        # running_acc = []
        # curr_loss = torch.zeros(1).to(self.device)
        # curr_acc = 0
        # for epoch in range(num_of_epochs):
        #     for idx, (data, label) in enumerate(train_loader):
        #         data, label = data.to(self.device), label.to(self.device)
        #         data, label = data.squeeze().T.reshape(-1, 1), label.reshape(-1, 1)
        #
        #         out = self._forward(data)
        #         loss = self._criterion(label, out)
        #         self._backward()
        #         self._optimizer(learning_rate)
        #         curr_loss += loss
        #
        #         if idx == 0 or (idx + 1) % 10000 == 0:
        #             if idx == 0:
        #                 running_loss.append(curr_loss.item())
        #             else:
        #                 running_loss.append((curr_loss / 10000).item())
        #             test_idx = 1
        #             for test_data, test_label in test_loader:
        #                 test_data, test_label = test_data.to(self.device), test_label.to(self.device)
        #                 test_data = test_data.squeeze().T.reshape(-1, 1)
        #                 test_out = self._forward(test_data, mode='eval')
        #                 pred_label = torch.argmax(test_out)
        #                 if pred_label == test_label:
        #                     curr_acc = curr_acc + 1
        #                 test_idx = test_idx + 1
        #             running_acc.append(curr_acc / (test_idx + 1))
        #             print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
        #             curr_loss = torch.zeros(1).to(self.device)
        #             curr_acc = 0
