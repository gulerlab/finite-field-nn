# extra utils
import logging
from abc import ABC, abstractmethod
import math

# torch
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16_bn, VGG16_BN_Weights

# this project
from utils import to_real_domain, to_finite_field_domain, finite_field_truncation, ToFiniteFieldDomain, \
    to_finite_field_domain_int, to_int_domain, ToIntDomain, from_int_to_real_domain, to_int_domain_int, \
    finite_field_truncation_ext


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


# noinspection DuplicatedCode
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


# noinspection DuplicatedCode
class VectorizedNet(AbstractVectorizedNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None

    @property
    def running_loss(self):
        return self.__running_loss

    @running_loss.setter
    def running_loss(self, value):
        self.__running_loss = value

    @property
    def running_acc(self):
        return self.__running_acc

    @running_acc.setter
    def running_acc(self, value):
        self.__running_acc = value

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

                if idx == 0 or (idx + 1) % 100 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss.item())
                    else:
                        running_loss.append((curr_loss / 100).item())
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
                    if idx == 0 or (idx + 1) % 10000 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc


# noinspection DuplicatedCode
class ScaledVectorizedNet(AbstractVectorizedNet):
    def __init__(self, scale_input_parameter, scale_weight_parameter, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None

        self.__scale_input_parameter = 2 ** scale_input_parameter
        self.__scale_weight_parameter = 2 ** scale_weight_parameter

        self.__scale_init_weight()

    def __scale_init_weight(self):
        self._weight_1 = self._weight_1 * self.__scale_weight_parameter
        self._weight_2 = self._weight_2 * self.__scale_weight_parameter

    @property
    def running_loss(self):
        return self.__running_loss

    @running_loss.setter
    def running_loss(self, value):
        self.__running_loss = value

    @property
    def running_acc(self):
        return self.__running_acc

    @running_acc.setter
    def running_acc(self, value):
        self.__running_acc = value

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

    def _criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        self.__save_for_backward['label'] = label
        # label, prediction = label / self.__scale_weight_parameter, prediction / self.__scale_weight_parameter
        diff = label - prediction
        diff_norm = torch.linalg.norm(diff)
        return torch.square(diff_norm)

    def _optimizer(self, learning_rate: float):
        self._weight_2 = self._weight_2 - learning_rate * self.__gradients['weight_2_grad']
        self._weight_1 = self._weight_1 - learning_rate * self.__gradients['weight_1_grad']

    def _forward(self, input_vector: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        first_forward = torch.square(torch.matmul(torch.t(self._weight_1),
                                                  input_vector)) / ((self.__scale_input_parameter ** 2) *
                                                                    self.__scale_weight_parameter)
        out = torch.matmul(torch.t(self._weight_2), first_forward) / self.__scale_weight_parameter
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

        weight_2_grad = -2 * torch.matmul(first_forward, torch.t(label - out)) / self.__scale_weight_parameter

        # weight_1 gradients
        second_chain = 2 * torch.diag(torch.matmul(torch.t(self._weight_1), input_vector)
                                      .reshape(-1)) / self.__scale_input_parameter
        third_chain = torch.t(self._weight_2)
        fourth_chain = -2 * torch.t(label - out)
        weight_1_grad = second_chain
        weight_1_grad = torch.matmul(third_chain, weight_1_grad) / self.__scale_weight_parameter
        weight_1_grad = torch.matmul(fourth_chain, weight_1_grad) / self.__scale_weight_parameter
        weight_1_grad = torch.matmul(input_vector, weight_1_grad) / self.__scale_input_parameter

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    # noinspection DuplicatedCode
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

                data, label = data * self.__scale_input_parameter, label * self.__scale_weight_parameter

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss

                if idx == 0 or (idx + 1) % 1000 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss.item())
                    else:
                        running_loss.append((curr_loss / 1000).item())
                    test_idx = 1
                    for test_data, test_label in test_loader:
                        test_data, test_label = test_data.to(self.device), test_label.to(self.device)
                        test_data = test_data.squeeze().T.reshape(-1, 1) * self.__scale_input_parameter
                        test_out = self._forward(test_data, mode='eval') / self.__scale_weight_parameter
                        pred_label = torch.argmax(test_out)
                        if pred_label == test_label:
                            curr_acc = curr_acc + 1
                        test_idx = test_idx + 1
                    running_acc.append(curr_acc / (test_idx + 1))
                    if idx == 0 or (idx + 1) % 1000 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc


# noinspection DuplicatedCode
class ScaledVectorizedIntegerNet(AbstractVectorizedNet):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None

        self.__scale_input_parameter = scale_input_parameter
        self.__scale_weight_parameter = scale_weight_parameter
        self.__scale_learning_rate_parameter = scale_learning_rate_parameter

        self.__scale_init_weight()

        self.__max_weight_value_1 = torch.max(self._weight_1)
        self.__min_weight_value_1 = torch.min(self._weight_1)

        self.__max_weight_value_2 = torch.max(self._weight_2)
        self.__min_weight_value_2 = torch.min(self._weight_2)

        self.__max_input_value, self.__min_input_value = None, None

    def __scale_init_weight(self):
        self._weight_1 = to_int_domain(self._weight_1, self.__scale_weight_parameter)
        self._weight_2 = to_int_domain(self._weight_2, self.__scale_weight_parameter)

    @property
    def running_loss(self):
        return self.__running_loss

    @running_loss.setter
    def running_loss(self, value):
        self.__running_loss = value

    @property
    def running_acc(self):
        return self.__running_acc

    @running_acc.setter
    def running_acc(self, value):
        self.__running_acc = value

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
    def scale_learning_rate_parameter(self):
        return self.__scale_learning_rate_parameter

    @scale_learning_rate_parameter.setter
    def scale_learning_rate_parameter(self, value):
        self.__scale_learning_rate_parameter = value

    @property
    def min_max_weight_parameter(self):
        return self.__min_weight_value_1.item(), self.__min_weight_value_2.item(), self.__max_weight_value_1.item(), \
            self.__max_weight_value_2.item()

    @min_max_weight_parameter.setter
    def min_max_weight_parameter(self, tuple_min_max):
        self.__min_weight_value_1, self.__min_weight_value_2, self.__max_weight_value_1, \
            self.__max_weight_value_2 = tuple_min_max

    @property
    def min_max_input_parameter(self):
        return self.__min_input_value.item(), self.__max_input_value.item()

    @min_max_input_parameter.setter
    def min_max_input_parameter(self, tuple_min_max):
        self.__min_input_value, self.__max_input_value = tuple_min_max

    def _criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        self.__save_for_backward['label'] = label
        real_label = from_int_to_real_domain(label, self.__scale_weight_parameter)
        real_prediction = from_int_to_real_domain(prediction, self.__scale_weight_parameter)
        diff = real_label - real_prediction
        diff_norm = torch.linalg.norm(diff)
        return torch.square(diff_norm)

    def _optimizer(self, learning_rate: float):
        learning_rate = to_int_domain_int(learning_rate, self.__scale_learning_rate_parameter)
        weight_2_grad = finite_field_truncation(learning_rate * self.__gradients['weight_2_grad'],
                                                self.__scale_learning_rate_parameter)
        weight_1_grad = finite_field_truncation(learning_rate * self.__gradients['weight_1_grad'],
                                                self.__scale_learning_rate_parameter)
        self._weight_2 = self._weight_2 - weight_2_grad
        self._weight_1 = self._weight_1 - weight_1_grad

        self.__max_weight_value_1 = torch.max(self._weight_1)
        self.__min_weight_value_1 = torch.min(self._weight_1)

        self.__max_weight_value_2 = torch.max(self._weight_2)
        self.__min_weight_value_2 = torch.min(self._weight_2)

    def _forward(self, input_vector: torch.Tensor, mode: str = 'train') -> torch.Tensor:

        first_forward = finite_field_truncation(torch.matmul(torch.t(self._weight_1).type(torch.float),
                                                             input_vector.type(torch.float)),
                                                self.__scale_input_parameter)
        first_forward = finite_field_truncation(torch.square(first_forward.type(torch.float)),
                                                self.__scale_weight_parameter)
        out = finite_field_truncation(torch.matmul(torch.t(self._weight_2).type(torch.float),
                                                   first_forward.type(torch.float)), self.__scale_weight_parameter)
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

        weight_2_grad = -2 * finite_field_truncation(torch.matmul(first_forward.type(torch.float),
                                                                  torch.t(label - out).type(torch.float)),
                                                     self.__scale_weight_parameter)

        # weight_1 gradients
        second_chain = 2 * finite_field_truncation(torch.diag(torch.matmul(torch.t(self._weight_1).type(torch.float),
                                                                           input_vector.type(torch.float)).reshape(-1)),
                                                   self.__scale_input_parameter)
        third_chain = torch.t(self._weight_2)
        fourth_chain = -2 * torch.t(label - out)
        weight_1_grad = second_chain
        weight_1_grad = finite_field_truncation(torch.matmul(third_chain.type(torch.float),
                                                             weight_1_grad.type(torch.float)),
                                                self.__scale_weight_parameter)
        weight_1_grad = finite_field_truncation(torch.matmul(fourth_chain.type(torch.float),
                                                             weight_1_grad.type(torch.float)),
                                                self.__scale_weight_parameter)
        weight_1_grad = finite_field_truncation(torch.matmul(input_vector.type(torch.float),
                                                             weight_1_grad.type(torch.float)),
                                                self.__scale_input_parameter)

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    # noinspection DuplicatedCode
    def train(self, data_path: str, num_of_epochs: int, learning_rate: float):
        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            ToIntDomain(self.__scale_input_parameter)
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1)),
            ToIntDomain(self.__scale_weight_parameter)
        ])

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

                if self.__max_input_value is None:
                    self.__max_input_value = torch.max(data)
                else:
                    if self.__max_input_value < torch.max(data):
                        self.__max_input_value = torch.max(data)

                if self.__min_input_value is None:
                    self.__min_input_value = torch.min(data)
                else:
                    if self.__min_input_value < torch.min(data):
                        self.__min_input_value = torch.min(data)

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss

                if idx == 0 or (idx + 1) % 100 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss.item())
                    else:
                        running_loss.append((curr_loss / 100).item())
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
                    if idx == 0 or (idx + 1) % 100 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        torch.save({
            'model': {
                'weight_1': self._weight_1,
                'weight_2': self._weight_2
            },
            'min_weight_1': self.__min_weight_value_1,
            'max_weight_1': self.__max_weight_value_1,
            'min_weight_2': self.__min_weight_value_2,
            'max_weight_2': self.__max_weight_value_2,
            'min_input': self.__min_input_value,
            'max_input': self.__max_input_value,
            'running_loss': self.__running_loss,
            'running_acc': self.__running_loss
        }, 'params/scaled_vectorized_int_nn_params.tar.gz')


# noinspection DuplicatedCode
class ScaledVectorizedFiniteFieldNet(AbstractVectorizedNet):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, prime, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None

        self.__scale_input_parameter = scale_input_parameter
        self.__scale_weight_parameter = scale_weight_parameter
        self.__scale_learning_rate_parameter = scale_learning_rate_parameter
        self.__prime = prime

        self.__scale_init_weight()

    def __scale_init_weight(self):
        self._weight_1 = to_finite_field_domain(self._weight_1, self.__scale_weight_parameter, self.__prime)
        self._weight_2 = to_finite_field_domain(self._weight_2, self.__scale_weight_parameter, self.__prime)

    @property
    def running_loss(self):
        return self.__running_loss

    @running_loss.setter
    def running_loss(self, value):
        self.__running_loss = value

    @property
    def running_acc(self):
        return self.__running_acc

    @running_acc.setter
    def running_acc(self, value):
        self.__running_acc = value

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
    def scale_learning_rate_parameter(self):
        return self.__scale_learning_rate_parameter

    @scale_learning_rate_parameter.setter
    def scale_learning_rate_parameter(self, value):
        self.__scale_learning_rate_parameter = value

    @property
    def prime(self):
        return self.__prime

    @prime.setter
    def prime(self, value):
        self.__prime = value

    def _criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        self.__save_for_backward['label'] = label
        real_label = to_real_domain(label, self.__scale_weight_parameter, self.__prime)
        real_prediction = to_real_domain(prediction, self.__scale_weight_parameter, self.__prime)
        diff = real_label - real_prediction
        diff_norm = torch.linalg.norm(diff)
        return torch.square(diff_norm)

    def _optimizer(self, learning_rate: float):
        learning_rate = to_finite_field_domain_int(learning_rate, self.__scale_learning_rate_parameter, self.__prime)
        weight_2_grad = finite_field_truncation_ext(learning_rate * self.__gradients['weight_2_grad'] % self.__prime,
                                                    self.__scale_learning_rate_parameter, self.__prime)
        weight_1_grad = finite_field_truncation_ext(learning_rate * self.__gradients['weight_1_grad'] % self.__prime,
                                                    self.__scale_learning_rate_parameter, self.__prime)
        self._weight_2 = (self._weight_2 - weight_2_grad) % self.__prime
        self._weight_1 = (self._weight_1 - weight_1_grad) % self.__prime

    def _forward(self, input_vector: torch.Tensor, mode: str = 'train') -> torch.Tensor:

        first_forward = finite_field_truncation_ext(torch.matmul(torch.t(self._weight_1).type(torch.float),
                                                                 input_vector.type(torch.float)) % self.__prime,
                                                    self.__scale_input_parameter, self.__prime)
        first_forward = finite_field_truncation_ext(torch.square(first_forward.type(torch.float)) % self.__prime,
                                                    self.__scale_weight_parameter, self.__prime)
        out = finite_field_truncation_ext(torch.matmul(torch.t(self._weight_2).type(torch.float),
                                                       first_forward.type(torch.float)) % self.__prime,
                                          self.__scale_weight_parameter, self.__prime)
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

        weight_2_grad = ((self.__prime - 2) * finite_field_truncation_ext(torch.matmul(first_forward.type(torch.float),
                                                                                       torch.t((label - out) %
                                                                                               self.__prime)
                                                                                       .type(torch.float))
                                                                          % self.__prime,
                                                                          self.__scale_weight_parameter,
                                                                          self.__prime)) % self.__prime

        # weight_1 gradients
        second_chain = (2 * finite_field_truncation_ext(torch.diag(torch.matmul(torch.t(self._weight_1)
                                                                                .type(torch.float),
                                                                                input_vector.type(torch.float))
                                                                   .reshape(-1)) % self.__prime,
                                                        self.__scale_input_parameter, self.__prime)) % self.__prime
        third_chain = torch.t(self._weight_2)
        fourth_chain = ((self.__prime - 2) * torch.t((label - out) % self.__prime)) % self.__prime
        weight_1_grad = second_chain
        weight_1_grad = finite_field_truncation_ext(torch.matmul(third_chain.type(torch.float),
                                                                 weight_1_grad.type(torch.float)) % self.__prime,
                                                    self.__scale_weight_parameter, self.__prime)
        weight_1_grad = finite_field_truncation_ext(torch.matmul(fourth_chain.type(torch.float),
                                                                 weight_1_grad.type(torch.float)) % self.__prime,
                                                    self.__scale_weight_parameter, self.__prime)
        weight_1_grad = finite_field_truncation_ext(torch.matmul(input_vector.type(torch.float),
                                                                 weight_1_grad.type(torch.float)) % self.__prime,
                                                    self.__scale_input_parameter, self.__prime)

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    # noinspection DuplicatedCode
    def train(self, data_path: str, num_of_epochs: int, learning_rate: float):
        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            ToFiniteFieldDomain(self.__scale_input_parameter, self.__prime)
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1)),
            ToFiniteFieldDomain(self.__scale_weight_parameter, self.__prime)
        ])

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

                if idx == 0 or (idx + 1) % 100 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss.item())
                    else:
                        running_loss.append((curr_loss / 100).item())
                    test_idx = 1
                    for test_data, test_label in test_loader:
                        test_data, test_label = test_data.to(self.device), test_label.to(self.device)
                        test_data = test_data.squeeze().T.reshape(-1, 1)
                        test_out = self._forward(test_data, mode='eval')
                        test_out = to_real_domain(test_out, self.__scale_weight_parameter, self.__prime)
                        pred_label = torch.argmax(test_out)
                        if pred_label == test_label:
                            curr_acc = curr_acc + 1
                        test_idx = test_idx + 1
                    running_acc.append(curr_acc / (test_idx + 1))
                    if idx == 0 or (idx + 1) % 100 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc


# noinspection DuplicatedCode
class AbstractNet(ABC):
    def __init__(self, feature_size=784, hidden_layer_size=64, num_classes=10,
                 device=None, verbose=True):
        self.__feature_size = feature_size
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
        range_low, range_high = -1 / math.sqrt(self.__feature_size), 1 / math.sqrt(self.__feature_size)
        self._weight_1 = range_low + torch.rand((self.__feature_size, self.__hidden_layer_size)) * (range_high -
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
    def feature_size(self):
        return self.__feature_size

    @feature_size.setter
    def feature_size(self, value):
        self.__feature_size = value

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
    def train(self, data_path: str, num_of_epochs: int, learning_rate, batch_size: int):
        pass


# noinspection DuplicatedCode
class Net(AbstractNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_curr_loss = None
        self.__running_acc = None
        self.__batch_size = None

    @property
    def running_loss(self):
        return self.__running_loss

    @running_loss.setter
    def running_loss(self, value):
        self.__running_loss = value

    @property
    def running_curr_loss(self):
        return self.__running_curr_loss

    @running_curr_loss.setter
    def running_curr_loss(self, value):
        self.__running_curr_loss = value

    @property
    def running_acc(self):
        return self.__running_acc

    @running_acc.setter
    def running_acc(self, value):
        self.__running_acc = value

    @property
    def weight_1(self):
        return self._weight_1

    @property
    def weight_2(self):
        return self._weight_2

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value

    def _criterion(self, label: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        diff = label - prediction
        diff_norm = torch.linalg.norm(diff)
        self.__save_for_backward['label'] = label
        return torch.square(diff_norm) / prediction.size(0)

    def _optimizer(self, learning_rate: float):
        self._weight_2 = self._weight_2 - learning_rate * (self.__gradients['weight_2_grad'] / self.__batch_size)
        self._weight_1 = self._weight_1 - learning_rate * (self.__gradients['weight_1_grad'] / self.__batch_size)

    def _forward(self, input_matrix: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        before_activation = torch.matmul(input_matrix, self._weight_1)
        first_forward = torch.square(before_activation)
        out = torch.matmul(first_forward, self._weight_2)
        if mode == 'train':
            self.__save_for_backward = {
                'input_matrix': input_matrix,
                'before_activation': before_activation,
                'first_forward': first_forward,
                'out': out
            }

        return out

    def _backward(self):
        first_forward, out, label, input_matrix, before_activation = self.__save_for_backward['first_forward'], \
            self.__save_for_backward['out'], self.__save_for_backward['label'], \
            self.__save_for_backward['input_matrix'], self.__save_for_backward['before_activation']

        # weight_2 gradients
        weight_2_grad = -2 * torch.matmul(torch.t(first_forward), label - out)

        # weight_1 gradients
        middle_term = -2 * torch.matmul(label - out, torch.t(self._weight_2))
        last_term = 2 * middle_term * before_activation
        weight_1_grad = torch.matmul(torch.t(input_matrix), last_term)

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    def train(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size

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
        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        test_dataset = FashionMNIST(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.__batch_size, shuffle=True)

        running_loss = []
        running_acc = []
        for epoch in range(num_of_epochs):
            curr_loss = torch.zeros(1).to(self.device)
            curr_acc = 0
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                data, label = data.squeeze().reshape(data.size(0), -1), label.reshape(label.size(0), -1)

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss

                if idx == 0 or (idx + 1) % 10 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss.item())
                    else:
                        running_loss.append((curr_loss / 10).item())
                    test_total = 0
                    for test_data, test_label in test_loader:
                        test_data, test_label = test_data.to(self.device), test_label.to(self.device)
                        test_data = test_data.squeeze().reshape(test_data.size(0), -1)
                        test_out = self._forward(test_data, mode='eval')
                        pred_label = torch.argmax(test_out, dim=1)
                        curr_acc = curr_acc + torch.count_nonzero(pred_label == test_label)
                        test_total = test_total + test_data.size(0)
                    running_acc.append(curr_acc / test_total)
                    if idx == 0 or (idx + 1) % 10 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc

    def train_cifar10(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))
        # load data
        train_dataset = CIFAR10(data_path, train=True, transform=transform, target_transform=target_transform,
                                download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        test_dataset = CIFAR10(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.__batch_size, shuffle=True)

        last_batch_idx = int(len(train_dataset) / self.__batch_size)
        if len(train_dataset) % self.__batch_size != 0:
            last_batch_idx = last_batch_idx + 1

        running_loss = []
        running_acc = []
        running_curr_loss = []
        for epoch in range(num_of_epochs):
            curr_loss = torch.zeros(1).to(self.device)
            curr_acc = 0
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                data, label = data.squeeze().reshape(data.size(0), -1), label.reshape(label.size(0), -1)

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                running_curr_loss.append(loss.item())

                if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                    if idx == 0:
                        running_loss.append(curr_loss.item())
                    elif (idx + 1) == last_batch_idx:
                        running_loss.append((curr_loss / ((idx + 1) % 10)).item())
                    else:
                        running_loss.append((curr_loss / 10).item())
                    test_total = 0
                    for test_data, test_label in test_loader:
                        test_data, test_label = test_data.to(self.device), test_label.to(self.device)
                        test_data = test_data.squeeze().reshape(test_data.size(0), -1)
                        test_out = self._forward(test_data, mode='eval')
                        pred_label = torch.argmax(test_out, dim=1)
                        curr_acc = curr_acc + torch.count_nonzero(pred_label == test_label)
                        test_total = test_total + test_data.size(0)
                    running_acc.append(curr_acc / test_total)
                    if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        self.__running_curr_loss = running_curr_loss

    def train_vgg_cifar10(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))
        # load data
        train_dataset = CIFAR10(data_path, train=True, transform=transform, target_transform=target_transform,
                                download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        test_dataset = CIFAR10(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.__batch_size, shuffle=True)

        vgg_backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).eval()
        vgg_backbone = torch.nn.Sequential(*(list(vgg_backbone.children())[:-1])).to(device)

        last_batch_idx = int(len(train_dataset) / self.__batch_size)
        if len(train_dataset) % self.__batch_size != 0:
            last_batch_idx = last_batch_idx + 1

        with torch.no_grad():
            running_loss = []
            running_acc = []
            running_curr_loss = []
            for epoch in range(num_of_epochs):
                curr_loss = torch.zeros(1).to(self.device)
                curr_acc = 0
                for idx, (data, label) in enumerate(train_loader):
                    data = data.to(device)
                    data = vgg_backbone(data).reshape(data.size(0), -1).to(self.device)
                    label = label.to(self.device)

                    out = self._forward(data)
                    loss = self._criterion(label, out)
                    self._backward()
                    self._optimizer(learning_rate)
                    curr_loss += loss
                    running_curr_loss.append(loss.item())

                    if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                        if idx == 0:
                            running_loss.append(curr_loss.item())
                        elif (idx + 1) == last_batch_idx:
                            running_loss.append((curr_loss / ((idx + 1) % 10)).item())
                        else:
                            running_loss.append((curr_loss / 10).item())
                        test_total = 0
                        for test_data, test_label in test_loader:
                            test_data = test_data.to(device)
                            test_data = vgg_backbone(test_data).reshape(test_data.size(0), -1).to(self.device)
                            test_label = test_label.to(self.device)
                            test_out = self._forward(test_data, mode='eval')
                            pred_label = torch.argmax(test_out, dim=1)
                            curr_acc = curr_acc + torch.count_nonzero(pred_label == test_label)
                            test_total = test_total + test_data.size(0)
                            del test_data
                            del test_label
                        running_acc.append(curr_acc / test_total)
                        if idx == 0 or (idx + 1) % 10 == 0:
                            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        curr_loss = torch.zeros(1).to(self.device)
                        curr_acc = 0
            self.__running_loss = running_loss
            self.__running_acc = running_acc
            self.__running_curr_loss = running_curr_loss

    # NOT WORKING
    def train_vgg_cifar10_v2(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))
        # load data
        train_dataset = CIFAR10(data_path, train=True, transform=transform, target_transform=target_transform,
                                download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        test_dataset = CIFAR10(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.__batch_size, shuffle=True)

        vgg_backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        vgg_backbone_without_classifier = torch.nn.Sequential(*(list(vgg_backbone.children())[:-1])).to(device)
        first_classifier_layer = torch.nn.Sequential(*list(list(vgg_backbone.children())[-1]
                                                           .children())[:-4]).to(device)

        with torch.no_grad():
            running_loss = []
            running_acc = []
            running_curr_loss = []
            for epoch in range(num_of_epochs):
                curr_loss = torch.zeros(1).to(self.device)
                curr_acc = 0
                for idx, (data, label) in enumerate(train_loader):
                    data = data.to(device)
                    data = vgg_backbone_without_classifier(data).reshape(data.size(0), -1)
                    data = first_classifier_layer(data).to(self.device)
                    label = label.to(self.device)

                    out = self._forward(data)
                    loss = self._criterion(label, out)
                    self._backward()
                    self._optimizer(learning_rate)
                    curr_loss += loss
                    running_curr_loss.append(loss.item())

                    if idx == 0 or (idx + 1) % 10 == 0:
                        if idx == 0:
                            running_loss.append(curr_loss.item())
                        else:
                            running_loss.append((curr_loss / 10).item())
                        test_total = 0
                        for test_data, test_label in test_loader:
                            test_data = test_data.to(device)
                            test_data = vgg_backbone_without_classifier(test_data).reshape(test_data.size(0), -1)
                            test_data = first_classifier_layer(test_data).to(self.device)
                            test_label = test_label.to(self.device)
                            test_out = self._forward(test_data, mode='eval')
                            pred_label = torch.argmax(test_out, dim=1)
                            curr_acc = curr_acc + torch.count_nonzero(pred_label == test_label)
                            test_total = test_total + test_data.size(0)
                        running_acc.append(curr_acc / test_total)
                        if idx == 0 or (idx + 1) % 10 == 0:
                            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        curr_loss = torch.zeros(1).to(self.device)
                        curr_acc = 0
            self.__running_loss = running_loss
            self.__running_acc = running_acc
            self.__running_curr_loss = running_curr_loss
