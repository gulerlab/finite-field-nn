# utils
import copy
import random
import numpy as np

# torch
import torch
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# this project
from utils import to_real_domain, to_finite_field_domain, finite_field_truncation, ToFiniteFieldDomain, \
    to_finite_field_domain_int, to_int_domain, ToIntDomain, from_int_to_real_domain, to_int_domain_int, \
    finite_field_truncation_ext

from simple_network import AbstractVectorizedNet


# noinspection DuplicatedCode
class ScaledVectorizedIntegerNet(AbstractVectorizedNet):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter,
                 save_num_of_iterations=3, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None

        self.__scale_input_parameter = scale_input_parameter
        self.__scale_weight_parameter = scale_weight_parameter
        self.__scale_learning_rate_parameter = scale_learning_rate_parameter

        self.__save_num_of_iterations = save_num_of_iterations
        self.__saved_data = torch.zeros((self.__save_num_of_iterations, self.input_vector_size), dtype=torch.long)
        self.__saved_weight_1 = torch.zeros((self.__save_num_of_iterations + 1, self._weight_1.shape[0],
                                             self._weight_1.shape[1]), dtype=torch.long)
        self.__saved_weight_2 = torch.zeros((self.__save_num_of_iterations + 1, self._weight_2.shape[0],
                                             self._weight_2.shape[1]), dtype=torch.long)
        self.__saved_label = torch.zeros((self.__save_num_of_iterations, self.num_classes), dtype=torch.long)
        self.__saved_gradients = []
        self.__saved_forward = []

        self.__scale_init_weight()

        self.__max_weight_value_1 = torch.max(self._weight_1)
        self.__min_weight_value_1 = torch.min(self._weight_1)

        self.__max_weight_value_2 = torch.max(self._weight_2)
        self.__min_weight_value_2 = torch.min(self._weight_2)

        self.__max_input_value, self.__min_input_value = None, None

    def __scale_init_weight(self):
        self._weight_1 = to_int_domain(self._weight_1, self.__scale_weight_parameter)
        self.__saved_weight_1[0] = copy.deepcopy(self._weight_1)
        self._weight_2 = to_int_domain(self._weight_2, self.__scale_weight_parameter)
        self.__saved_weight_2[0] = copy.deepcopy(self._weight_2)

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

        first_forward = finite_field_truncation(torch.matmul(torch.t(self._weight_1).type(torch.double),
                                                             input_vector.type(torch.double)),
                                                self.__scale_input_parameter)
        first_forward = finite_field_truncation(torch.square(first_forward.type(torch.double)),
                                                self.__scale_weight_parameter)
        out = finite_field_truncation(torch.matmul(torch.t(self._weight_2).type(torch.double),
                                                   first_forward.type(torch.double)), self.__scale_weight_parameter)
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

        weight_2_grad = -2 * finite_field_truncation(torch.matmul(first_forward.type(torch.double),
                                                                  torch.t(label - out).type(torch.double)),
                                                     self.__scale_weight_parameter)

        # weight_1 gradients
        second_chain = 2 * finite_field_truncation(torch.diag(torch.matmul(torch.t(self._weight_1).type(torch.double),
                                                                           input_vector.type(
                                                                               torch.double)).reshape(-1)),
                                                   self.__scale_input_parameter)
        third_chain = torch.t(self._weight_2)
        fourth_chain = -2 * torch.t(label - out)
        weight_1_grad = second_chain
        weight_1_grad = finite_field_truncation(torch.matmul(third_chain.type(torch.double),
                                                             weight_1_grad.type(torch.double)),
                                                self.__scale_weight_parameter)
        weight_1_grad = finite_field_truncation(torch.matmul(fourth_chain.type(torch.double),
                                                             weight_1_grad.type(torch.double)),
                                                self.__scale_weight_parameter)
        weight_1_grad = finite_field_truncation(torch.matmul(input_vector.type(torch.double),
                                                             weight_1_grad.type(torch.double)),
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
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.double)
                              .scatter_(0, torch.tensor(y), 1)),
            ToIntDomain(self.__scale_weight_parameter)
        ])

        generator = torch.Generator()
        generator.manual_seed(0)

        # load data
        train_dataset = FashionMNIST(data_path, train=True, transform=transform, target_transform=target_transform,
                                     download=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, generator=generator)

        for epoch in range(num_of_epochs):
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                data, label = data.squeeze().T.reshape(-1, 1), label.reshape(-1, 1)
                self.__saved_data[idx] = copy.deepcopy(data.reshape(-1))
                self.__saved_label[idx] = copy.deepcopy(label.reshape(-1))

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
                self._criterion(label, out)
                self.__saved_forward.append(copy.deepcopy(self.__save_for_backward))
                self._backward()
                self.__saved_gradients.append(copy.deepcopy(self.__gradients))
                self._optimizer(learning_rate)
                self.__saved_weight_1[idx + 1] = copy.deepcopy(self._weight_1)
                self.__saved_weight_2[idx + 1] = copy.deepcopy(self._weight_2)

                if idx == self.__save_num_of_iterations - 1:
                    break
            break

        torch.save({
            'data': self.__saved_data,
            'weight_1': self.__saved_weight_1,
            'weight_2': self.__saved_weight_2,
            'label': self.__saved_label,
            'gradients': self.__saved_gradients,
            'forward': self.__saved_forward
        }, 'params/{}_iterations_int.tar.gz'.format(self.__save_num_of_iterations))


# noinspection DuplicatedCode
class ScaledVectorizedFiniteFieldNet(AbstractVectorizedNet):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, prime,
                 save_num_of_iterations=3, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None

        self.__scale_input_parameter = scale_input_parameter
        self.__scale_weight_parameter = scale_weight_parameter
        self.__scale_learning_rate_parameter = scale_learning_rate_parameter
        self.__prime = prime

        self.__save_num_of_iterations = save_num_of_iterations
        self.__saved_data = torch.zeros((self.__save_num_of_iterations, self.input_vector_size), dtype=torch.long)
        self.__saved_weight_1 = torch.zeros((self.__save_num_of_iterations + 1, self._weight_1.shape[0],
                                             self._weight_1.shape[1]), dtype=torch.long)
        self.__saved_weight_2 = torch.zeros((self.__save_num_of_iterations + 1, self._weight_2.shape[0],
                                             self._weight_2.shape[1]), dtype=torch.long)
        self.__saved_label = torch.zeros((self.__save_num_of_iterations, self.num_classes), dtype=torch.long)
        self.__saved_gradients = []
        self.__saved_forward = []

        self.__scale_init_weight()

    def __scale_init_weight(self):
        self._weight_1 = to_finite_field_domain(self._weight_1, self.__scale_weight_parameter, self.__prime)
        self.__saved_weight_1[0] = copy.deepcopy(self._weight_1)
        self._weight_2 = to_finite_field_domain(self._weight_2, self.__scale_weight_parameter, self.__prime)
        self.__saved_weight_2[0] = copy.deepcopy(self._weight_2)

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

        first_forward = finite_field_truncation_ext(torch.matmul(torch.t(self._weight_1).type(torch.double),
                                                                 input_vector.type(torch.double)).type(torch.long) % self.__prime,
                                                    self.__scale_input_parameter, self.__prime)
        first_forward = finite_field_truncation_ext(torch.square(first_forward.type(torch.double)) % self.__prime,
                                                    self.__scale_weight_parameter, self.__prime)
        out = finite_field_truncation_ext(torch.matmul(torch.t(self._weight_2).type(torch.double),
                                                       first_forward.type(torch.double)) % self.__prime,
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

        weight_2_grad = ((self.__prime - 2) * finite_field_truncation_ext(torch.matmul(first_forward.type(torch.double),
                                                                                       torch.t((label - out) %
                                                                                               self.__prime)
                                                                                       .type(torch.double))
                                                                          % self.__prime,
                                                                          self.__scale_weight_parameter,
                                                                          self.__prime)) % self.__prime

        # weight_1 gradients
        second_chain = (2 * finite_field_truncation_ext(torch.diag(torch.matmul(torch.t(self._weight_1)
                                                                                .type(torch.double),
                                                                                input_vector.type(torch.double))
                                                                   .reshape(-1)) % self.__prime,
                                                        self.__scale_input_parameter, self.__prime)) % self.__prime
        third_chain = torch.t(self._weight_2)
        fourth_chain = ((self.__prime - 2) * torch.t((label - out) % self.__prime)) % self.__prime
        weight_1_grad = second_chain
        weight_1_grad = finite_field_truncation_ext(torch.matmul(third_chain.type(torch.double),
                                                                 weight_1_grad.type(torch.double)) % self.__prime,
                                                    self.__scale_weight_parameter, self.__prime)
        weight_1_grad = finite_field_truncation_ext(torch.matmul(fourth_chain.type(torch.double),
                                                                 weight_1_grad.type(torch.double)) % self.__prime,
                                                    self.__scale_weight_parameter, self.__prime)
        weight_1_grad = finite_field_truncation_ext(torch.matmul(input_vector.type(torch.double),
                                                                 weight_1_grad.type(torch.double)) % self.__prime,
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
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.double)
                              .scatter_(0, torch.tensor(y), 1)),
            ToFiniteFieldDomain(self.__scale_weight_parameter, self.__prime)
        ])

        generator = torch.Generator()
        generator.manual_seed(0)

        # load data
        train_dataset = FashionMNIST(data_path, train=True, transform=transform, target_transform=target_transform,
                                     download=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, generator=generator)

        for epoch in range(num_of_epochs):
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                data, label = data.squeeze().T.reshape(-1, 1), label.reshape(-1, 1)
                self.__saved_data[idx] = copy.deepcopy(data.reshape(-1))
                self.__saved_label[idx] = copy.deepcopy(label.reshape(-1))

                out = self._forward(data)
                self._criterion(label, out)
                self.__saved_forward.append(copy.deepcopy(self.__save_for_backward))
                self._backward()
                self.__saved_gradients.append(copy.deepcopy(self.__gradients))
                self._optimizer(learning_rate)
                self.__saved_weight_1[idx + 1] = copy.deepcopy(self._weight_1)
                self.__saved_weight_2[idx + 1] = copy.deepcopy(self._weight_2)

                if idx == self.__save_num_of_iterations - 1:
                    break
            break

        torch.save({
            'data': self.__saved_data,
            'weight_1': self.__saved_weight_1,
            'weight_2': self.__saved_weight_2,
            'label': self.__saved_label,
            'gradients': self.__saved_gradients,
            'forward': self.__saved_forward
        }, 'params/{}_iterations_ff.tar.gz'.format(self.__save_num_of_iterations))


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    int_net = ScaledVectorizedIntegerNet(8, 8, 10, save_num_of_iterations=1, device='cpu')
    int_net.train('./data', 1, 0.001)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    ff_net = ScaledVectorizedFiniteFieldNet(8, 8, 10, 2 ** 26 - 5, save_num_of_iterations=1, device='cpu')
    ff_net.train('./data', 1, 0.001)
