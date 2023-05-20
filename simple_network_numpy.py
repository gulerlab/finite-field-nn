# extra utils
from abc import ABC, abstractmethod
import math

# numpy
import numpy as np

# torch
import torch
from torchvision.datasets import FashionMNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16_bn, VGG16_BN_Weights

# this project
## int training
from utils_numpy import to_int_domain, ToIntDomain, from_int_to_real_domain, int_truncation, ToNumpy

## ff training
from utils_numpy import to_real_domain, to_finite_field_domain, finite_field_truncation, load_all_data, \
    from_finite_field_to_int_domain, create_batch_data, load_all_data_cifar10, load_all_data_apply_vgg_cifar10

## logging
from utils_numpy import info


# noinspection DuplicatedCode
class AbstractVectorizedNetNumpy(ABC):
    def __init__(self, input_vector_size=784, hidden_layer_size=64, num_classes=10):
        self.__input_vector_size = input_vector_size
        self.__hidden_layer_size = hidden_layer_size
        self.__num_classes = num_classes
        self._weight_1 = None
        self._weight_2 = None

        self.__init_weight()

    def __init_weight(self):
        range_low, range_high = -1 / math.sqrt(self.__input_vector_size), 1 / math.sqrt(self.__input_vector_size)
        self._weight_1 = range_low + np.random.rand(self.__input_vector_size, self.__hidden_layer_size) * (range_high
                                                                                                           - range_low)
        range_low, range_high = -1 / math.sqrt(self.__hidden_layer_size), 1 / math.sqrt(self.__hidden_layer_size)
        self._weight_2 = range_low + np.random.rand(self.__hidden_layer_size, self.__num_classes) * (range_high
                                                                                                     - range_low)
        info('weights are initialized')

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

    @abstractmethod
    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _optimizer(self, learning_rate):
        pass

    @abstractmethod
    def _forward(self, input_vector: np.ndarray, mode: str = 'train') -> np.ndarray:
        pass

    @abstractmethod
    def _backward(self):
        pass

    @abstractmethod
    def train(self, data_path: str, num_of_epochs: int, learning_rate):
        pass


# noinspection DuplicatedCode
class ScaledVectorizedIntegerNetNumpy(AbstractVectorizedNetNumpy):
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

    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> np.float64:
        self.__save_for_backward['label'] = label
        real_label = from_int_to_real_domain(label, self.__scale_weight_parameter)
        assert real_label.dtype == np.float64, 'ground truth labels are not in real domain in loss'
        real_prediction = from_int_to_real_domain(prediction, self.__scale_weight_parameter)
        assert real_label.dtype == np.float64, 'predictions are not in real domain in loss'
        diff = real_label - real_prediction
        diff_norm = np.linalg.norm(diff)
        return np.square(diff_norm)

    def _optimizer(self, learning_rate: float):
        weight_2_grad = int_truncation(self.__gradients['weight_2_grad'], self.__scale_learning_rate_parameter)
        weight_1_grad = int_truncation(self.__gradients['weight_1_grad'], self.__scale_learning_rate_parameter)
        self._weight_2 = self._weight_2 - weight_2_grad
        self._weight_1 = self._weight_1 - weight_1_grad

    def _forward(self, input_vector: np.ndarray, mode: str = 'train') -> np.ndarray:
        first_forward = int_truncation(self._weight_1.T @ input_vector, self.__scale_input_parameter)
        assert first_forward.dtype == np.int64, 'first forward before activation is not defined in int domain'
        first_forward = int_truncation(np.square(first_forward), self.__scale_weight_parameter)
        assert first_forward.dtype == np.int64, 'first forward after activation is not defined in int domain'
        out = int_truncation(self._weight_2.T @ first_forward, self.__scale_weight_parameter)
        assert out.dtype == np.int64, 'out is not defined in int domain'

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

        weight_2_grad = -2 * int_truncation(first_forward @ (label - out).T, self.__scale_weight_parameter)
        assert weight_2_grad.dtype == np.int64, 'gradient for second weight matrix is not defined in int domain'

        # weight_1 gradients
        second_chain = 2 * int_truncation(np.diag((self._weight_1.T @ input_vector).reshape(-1)),
                                          self.__scale_input_parameter)
        assert second_chain.dtype == np.int64, 'second chain is not defined in int domain'
        third_chain = self._weight_2.T
        assert third_chain.dtype == np.int64, 'third chain is not defined in int domain'
        fourth_chain = -2 * (label - out).T
        assert fourth_chain.dtype == np.int64, 'fourth chain is not defined in int domain'
        weight_1_grad = second_chain
        weight_1_grad = int_truncation(third_chain @ weight_1_grad, self.__scale_weight_parameter)
        weight_1_grad = int_truncation(fourth_chain @ weight_1_grad, self.__scale_weight_parameter)
        weight_1_grad = int_truncation(input_vector @ weight_1_grad, self.__scale_input_parameter)

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
            ToNumpy(),
            ToIntDomain(self.__scale_input_parameter)
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1)),
            ToNumpy(),
            ToIntDomain(self.__scale_weight_parameter)
        ])

        # load data
        train_dataset = FashionMNIST(data_path, train=True, transform=transform, target_transform=target_transform,
                                     download=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        test_dataset = FashionMNIST(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        curr_loss = 0
        curr_acc = 0
        for epoch in range(num_of_epochs):
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.squeeze().T.reshape((-1, 1)).numpy(), label.reshape((-1, 1)).numpy()

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss), verbose=False)

                if idx == 0 or (idx + 1) % 100 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 100))
                    test_idx = 1
                    for test_data, test_label in test_loader:
                        test_data = test_data.squeeze().T.reshape((-1, 1)).numpy()
                        test_out = self._forward(test_data, mode='eval')
                        pred_label = np.argmax(test_out)
                        if pred_label == test_label:
                            curr_acc = curr_acc + 1
                        test_idx = test_idx + 1
                    running_acc.append(curr_acc / (test_idx + 1))
                    if idx == 0 or (idx + 1) % 100 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                 running_loss[-1],
                                                                                                 running_acc[-1]),
                             verbose=False)
                    curr_loss = 0
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc


# noinspection DuplicatedCode
class ScaledVectorizedFiniteFieldNetNumpy(AbstractVectorizedNetNumpy):
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

    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> np.float64:
        self.__save_for_backward['label'] = label
        real_label = to_real_domain(label, self.__scale_weight_parameter, self.__prime)
        assert real_label.dtype == np.float64, 'ground truth labels are not in real domain in loss'
        real_prediction = to_real_domain(prediction, self.__scale_weight_parameter, self.__prime)
        assert real_label.dtype == np.float64, 'predictions are not in real domain in loss'
        diff = real_label - real_prediction
        diff_norm = np.linalg.norm(diff)
        return np.square(diff_norm)

    def _optimizer(self, learning_rate: float):
        weight_2_grad = finite_field_truncation(self.__gradients['weight_2_grad'], self.__scale_learning_rate_parameter,
                                                self.__prime)
        weight_1_grad = finite_field_truncation(self.__gradients['weight_1_grad'], self.__scale_learning_rate_parameter,
                                                self.__prime)

        weight_2_mask = self._weight_2 < weight_2_grad
        weight_2_diff_weight_2_grad = np.zeros(self._weight_2.shape, dtype=np.uint64)
        weight_2_diff_weight_2_grad[weight_2_mask] = (-1 * (weight_2_grad[weight_2_mask] -
                                                            self._weight_2[weight_2_mask])
                                                      .astype(np.int64)) % self.__prime
        weight_2_diff_weight_2_grad[~weight_2_mask] = self._weight_2[~weight_2_mask] - weight_2_grad[~weight_2_mask]

        weight_1_mask = self._weight_1 < weight_1_grad
        weight_1_diff_weight_1_grad = np.zeros(self._weight_1.shape, dtype=np.uint64)
        weight_1_diff_weight_1_grad[weight_1_mask] = (-1 * (weight_1_grad[weight_1_mask] -
                                                            self._weight_1[weight_1_mask])
                                                      .astype(np.int64)) % self.__prime
        weight_1_diff_weight_1_grad[~weight_1_mask] = self._weight_1[~weight_1_mask] - weight_1_grad[~weight_1_mask]

        self._weight_2 = weight_2_diff_weight_2_grad
        self._weight_1 = weight_1_diff_weight_1_grad

    def _forward(self, input_vector: np.ndarray, mode: str = 'train') -> np.ndarray:
        first_forward = finite_field_truncation((self._weight_1.T @ input_vector) % self.__prime,
                                                self.__scale_input_parameter, self.__prime)
        assert first_forward.dtype == np.uint64, 'first forward before activation is not defined in finite field domain'
        first_forward = finite_field_truncation(np.square(first_forward) % self.__prime, self.__scale_weight_parameter,
                                                self.__prime)
        assert first_forward.dtype == np.uint64, 'first forward after activation is not defined in finite field domain'
        out = finite_field_truncation((self._weight_2.T @ first_forward) % self.__prime, self.__scale_weight_parameter,
                                      self.__prime)
        assert out.dtype == np.uint64, 'out is not defined in finite field domain'

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

        label_mask = label < out
        label_diff_out = np.zeros(label.shape, dtype=np.uint64)
        label_diff_out[label_mask] = (-1 * (out[label_mask] - label[label_mask]).astype(np.int64)) % self.__prime
        label_diff_out[~label_mask] = label[~label_mask] - out[~label_mask]

        weight_2_grad = ((-2 % self.__prime) * finite_field_truncation((first_forward @ label_diff_out.T) %
                                                                       self.__prime, self.__scale_weight_parameter,
                                                                       self.__prime)) % self.__prime
        assert weight_2_grad.dtype == np.uint64, 'gradient for second weight matrix is not defined in finite' \
                                                 ' field domain'

        # weight_1 gradients
        second_chain = (2 * finite_field_truncation(np.diag((self._weight_1.T @ input_vector).reshape(-1)) %
                                                    self.__prime, self.__scale_input_parameter,
                                                    self.__prime)) % self.__prime
        assert second_chain.dtype == np.uint64, 'second chain is not defined in finite field domain'
        third_chain = self._weight_2.T
        assert third_chain.dtype == np.uint64, 'third chain is not defined in finite field domain'
        fourth_chain = ((-2 % self.__prime) * label_diff_out.T) % self.__prime
        assert fourth_chain.dtype == np.uint64, 'fourth chain is not defined in finite field domain'
        weight_1_grad = second_chain
        weight_1_grad = finite_field_truncation((third_chain @ weight_1_grad) % self.__prime,
                                                self.__scale_weight_parameter, self.__prime)
        weight_1_grad = finite_field_truncation((fourth_chain @ weight_1_grad) % self.__prime,
                                                self.__scale_weight_parameter, self.__prime)
        weight_1_grad = finite_field_truncation((input_vector @ weight_1_grad) % self.__prime,
                                                self.__scale_input_parameter, self.__prime)

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    # noinspection DuplicatedCode
    def train(self, data_path: str, num_of_epochs: int, learning_rate: float):
        train_data, train_label, test_data_all, test_label_all = load_all_data(self.__scale_input_parameter,
                                                                               self.__scale_weight_parameter,
                                                                               self.__prime)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        curr_loss = 0
        curr_acc = 0
        for epoch in range(num_of_epochs):
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                data, label = data.T.reshape((-1, 1)), label.reshape((-1, 1))

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss), verbose=False)

                if idx == 0 or (idx + 1) % 100 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 100))
                    test_idx = 1
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_data = test_data.T.reshape((-1, 1))
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self.__prime)
                        pred_label = np.argmax(test_out)
                        if pred_label == test_label:
                            curr_acc = curr_acc + 1
                        test_idx = test_idx + 1
                    running_acc.append(curr_acc / (test_idx + 1))
                    if idx == 0 or (idx + 1) % 100 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                 running_loss[-1],
                                                                                                 running_acc[-1]),
                             verbose=False)
                    curr_loss = 0
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc


# noinspection DuplicatedCode
class AbstractNetNumpy(ABC):
    def __init__(self, input_vector_size=784, hidden_layer_size=64, num_classes=10):
        self.__input_vector_size = input_vector_size
        self.__hidden_layer_size = hidden_layer_size
        self.__num_classes = num_classes
        self._weight_1 = None
        self._weight_2 = None

        self.__init_weight()

    def __init_weight(self):
        range_low, range_high = -1 / math.sqrt(self.__input_vector_size), 1 / math.sqrt(self.__input_vector_size)
        self._weight_1 = range_low + np.random.rand(self.__input_vector_size, self.__hidden_layer_size) * (range_high
                                                                                                           - range_low)
        range_low, range_high = -1 / math.sqrt(self.__hidden_layer_size), 1 / math.sqrt(self.__hidden_layer_size)
        self._weight_2 = range_low + np.random.rand(self.__hidden_layer_size, self.__num_classes) * (range_high
                                                                                                     - range_low)
        info('weights are initialized')

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

    @abstractmethod
    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _optimizer(self, learning_rate):
        pass

    @abstractmethod
    def _forward(self, input_vector: np.ndarray, mode: str = 'train') -> np.ndarray:
        pass

    @abstractmethod
    def _backward(self):
        pass

    @abstractmethod
    def train(self, data_path: str, num_of_epochs: int, learning_rate, batch_size: int):
        pass


# noinspection DuplicatedCode
class ScaledIntegerNetNumpy(AbstractNetNumpy):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None
        self.__running_curr_loss = None
        self.__batch_size = None
        self.__batch_size_param = None

        self.__scale_input_parameter = scale_input_parameter
        self.__scale_weight_parameter = scale_weight_parameter
        self.__scale_learning_rate_parameter = scale_learning_rate_parameter

        self.__scale_init_weight()

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
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value

    @property
    def batch_size_param(self):
        return self.__batch_size_param

    @batch_size_param.setter
    def batch_size_param(self, value):
        self.__batch_size_param = value

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
    def weight_1(self):
        return self._weight_1

    @property
    def weight_2(self):
        return self._weight_2

    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> np.float64:
        self.__save_for_backward['label'] = label
        real_label = from_int_to_real_domain(label, self.__scale_weight_parameter)
        assert real_label.dtype == np.float64, 'ground truth labels are not in real domain in loss'
        real_prediction = from_int_to_real_domain(prediction, self.__scale_weight_parameter)
        assert real_label.dtype == np.float64, 'predictions are not in real domain in loss'
        diff = real_label - real_prediction
        diff_norm = np.linalg.norm(diff)
        return np.square(diff_norm) / self.__batch_size

    def _optimizer(self, learning_rate: float):
        weight_2_grad = int_truncation(self.__gradients['weight_2_grad'], self.__scale_learning_rate_parameter
                                       + self.__batch_size_param)
        weight_1_grad = int_truncation(self.__gradients['weight_1_grad'], self.__scale_learning_rate_parameter
                                       + self.__batch_size_param)
        self._weight_2 = self._weight_2 - weight_2_grad
        self._weight_1 = self._weight_1 - weight_1_grad

    def _forward(self, input_matrix: np.ndarray, mode: str = 'train') -> np.ndarray:
        before_activation = int_truncation(input_matrix @ self._weight_1, self.__scale_input_parameter)
        assert before_activation.dtype == np.int64, 'before activation is not defined in int domain'
        first_forward = int_truncation(np.square(before_activation), self.__scale_weight_parameter)
        assert first_forward.dtype == np.int64, 'first forward after activation is not defined in int domain'
        out = int_truncation(first_forward @ self._weight_2, self.__scale_weight_parameter)
        assert out.dtype == np.int64, 'out is not defined in int domain'

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

        weight_2_grad = -2 * int_truncation(first_forward.T @ (label - out), self.__scale_weight_parameter)
        assert weight_2_grad.dtype == np.int64, 'gradient for second weight matrix is not defined in int domain'

        middle_term = -2 * int_truncation((label - out) @ self._weight_2.T, self.__scale_weight_parameter)
        assert middle_term.dtype == np.int64, 'middle term is not defined in int domain'
        last_term = 2 * int_truncation(middle_term * before_activation, self.__scale_weight_parameter)
        assert last_term.dtype == np.int64, 'last term is not defined in int domain'
        weight_1_grad = int_truncation(input_matrix.T @ last_term, self.__scale_input_parameter)
        assert weight_1_grad.dtype == np.int64, 'gradient for first weight matrix is not defined in int domain'

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    # noinspection DuplicatedCode
    def train(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        self.__batch_size_param = int(np.log2(self.__batch_size))

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            ToNumpy(),
            ToIntDomain(self.__scale_input_parameter)
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1)),
            ToNumpy(),
            ToIntDomain(self.__scale_weight_parameter)
        ])

        # load data
        train_dataset = FashionMNIST(data_path, train=True, transform=transform, target_transform=target_transform,
                                     download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        test_dataset = FashionMNIST(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.__batch_size, shuffle=True)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.squeeze().reshape((data.size(0), -1)).numpy(), label.reshape((label.size(0),
                                                                                                 -1)).numpy()

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss), verbose=False)

                if idx == 0 or (idx + 1) % 10 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in test_loader:
                        test_data = test_data.squeeze().reshape((test_data.size(0), -1)).numpy()
                        test_label = test_label.numpy()
                        test_out = self._forward(test_data, mode='eval')
                        pred_label = np.argmax(test_out, axis=1)
                        curr_acc = curr_acc + np.count_nonzero(pred_label == test_label)
                        test_total = test_total + test_data.shape[0]
                    running_acc.append(curr_acc / test_total)
                    if idx == 0 or (idx + 1) % 10 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                 running_loss[-1],
                                                                                                 running_acc[-1]),
                             verbose=False)
                    curr_loss = 0
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc

    # noinspection DuplicatedCode
    def train_cifar10(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        self.__batch_size_param = int(np.log2(self.__batch_size))

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ToNumpy(),
            ToIntDomain(self.__scale_input_parameter)
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1)),
            ToNumpy(),
            ToIntDomain(self.__scale_weight_parameter)
        ])

        # load data
        train_dataset = CIFAR10(data_path, train=True, transform=transform, target_transform=target_transform,
                                download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        test_dataset = CIFAR10(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.__batch_size, shuffle=True)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.squeeze().reshape((data.size(0), -1)).numpy(), label.reshape((label.size(0),
                                                                                                 -1)).numpy()

                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss), verbose=False)

                if idx == 0 or (idx + 1) % 10 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in test_loader:
                        test_data = test_data.squeeze().reshape((test_data.size(0), -1)).numpy()
                        test_label = test_label.numpy()
                        test_out = self._forward(test_data, mode='eval')
                        pred_label = np.argmax(test_out, axis=1)
                        curr_acc = curr_acc + np.count_nonzero(pred_label == test_label)
                        test_total = test_total + test_data.shape[0]
                    running_acc.append(curr_acc / test_total)
                    if idx == 0 or (idx + 1) % 10 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                 running_loss[-1],
                                                                                                 running_acc[-1]),
                             verbose=False)
                    curr_loss = 0
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc

    # noinspection DuplicatedCode
    def train_vgg_cifar10(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        self.__batch_size_param = int(np.log2(self.__batch_size))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1))
        ])

        # load data
        train_dataset = CIFAR10(data_path, train=True, transform=transform, target_transform=target_transform,
                                download=True)
        train_loader = DataLoader(train_dataset, batch_size=self.__batch_size, shuffle=True)

        test_dataset = CIFAR10(data_path, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=self.__batch_size, shuffle=True)
        last_batch_idx = int(len(train_dataset) / self.__batch_size)
        if len(train_dataset) % self.__batch_size != 0:
            last_batch_idx = last_batch_idx + 1

        info('datasets and loaders are initialized')

        vgg_backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).eval()
        vgg_backbone = torch.nn.Sequential(*(list(vgg_backbone.children())[:-1])).to(device)
        info('vgg backbone is loaded')

        with torch.no_grad():
            running_loss = []
            running_acc = []
            running_curr_loss = []
            for epoch in range(num_of_epochs):
                curr_loss = 0
                curr_acc = 0
                for idx, (data, label) in enumerate(train_loader):
                    data = data.to(device)
                    data = vgg_backbone(data).reshape(data.size(0), -1).to('cpu')
                    data = to_int_domain(data.numpy(), self.__scale_input_parameter)
                    label = to_int_domain(label.numpy(), self.__scale_weight_parameter)

                    out = self._forward(data)
                    loss = self._criterion(label, out)
                    self._backward()
                    self._optimizer(learning_rate)
                    curr_loss += loss
                    info('epoch: {}, iter: {}, loss: {}'.format(epoch, idx + 1, loss))
                    running_curr_loss.append(loss)

                    if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                        if idx == 0:
                            running_loss.append(curr_loss)
                        else:
                            running_loss.append((curr_loss / 10))
                        test_total = 0
                        for test_data, test_label in test_loader:
                            test_data = test_data.to(device)
                            test_data = vgg_backbone(test_data).reshape(test_data.size(0), -1).to('cpu')
                            test_data = to_int_domain(test_data.numpy(), self.__scale_input_parameter)
                            test_out = self._forward(test_data, mode='eval')
                            pred_label = np.argmax(test_out, axis=1)
                            test_label = test_label.numpy()
                            curr_acc = curr_acc + np.count_nonzero(pred_label == test_label)
                            test_total = test_total + test_data.shape[0]
                        running_acc.append(curr_acc / test_total)
                        if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                            info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                     running_loss[-1],
                                                                                                     running_acc[-1]),
                                 verbose=False)
                        curr_loss = 0
                        curr_acc = 0
            self.__running_loss = running_loss
            self.__running_acc = running_acc
            self.__running_curr_loss = running_curr_loss


# noinspection DuplicatedCode
class ScaledFiniteFieldNetNumpy(AbstractNetNumpy):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, prime, **kwargs):
        super().__init__(**kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None
        self.__running_curr_loss = None
        self.batch_size = None
        self.batch_size_param = None

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
    def running_curr_loss(self):
        return self.__running_curr_loss

    @running_curr_loss.setter
    def running_curr_loss(self, value):
        self.__running_curr_loss = value

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value

    @property
    def batch_size_param(self):
        return self.__batch_size_param

    @batch_size_param.setter
    def batch_size_param(self, value):
        self.__batch_size_param = value

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

    @property
    def weight_1(self):
        return self._weight_1

    @property
    def weight_2(self):
        return self._weight_2

    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> np.float64:
        self.__save_for_backward['label'] = label
        real_label = to_real_domain(label, self.__scale_weight_parameter, self.__prime)
        assert real_label.dtype == np.float64, 'ground truth labels are not in real domain in loss'
        real_prediction = to_real_domain(prediction, self.__scale_weight_parameter, self.__prime)
        assert real_label.dtype == np.float64, 'predictions are not in real domain in loss'
        diff = real_label - real_prediction
        diff_norm = np.linalg.norm(diff)
        return np.square(diff_norm) / self.__batch_size

    def _optimizer(self, learning_rate: float):
        weight_2_grad = finite_field_truncation(self.__gradients['weight_2_grad'], self.__scale_learning_rate_parameter
                                                + self.__batch_size_param, self.__prime)
        weight_1_grad = finite_field_truncation(self.__gradients['weight_1_grad'], self.__scale_learning_rate_parameter
                                                + self.__batch_size_param, self.__prime)

        weight_2_mask = self._weight_2 < weight_2_grad
        weight_2_diff_weight_2_grad = np.zeros(self._weight_2.shape, dtype=np.uint64)
        weight_2_diff_weight_2_grad[weight_2_mask] = (-1 * (weight_2_grad[weight_2_mask] -
                                                            self._weight_2[weight_2_mask])
                                                      .astype(np.int64)) % self.__prime
        weight_2_diff_weight_2_grad[~weight_2_mask] = self._weight_2[~weight_2_mask] - weight_2_grad[~weight_2_mask]

        weight_1_mask = self._weight_1 < weight_1_grad
        weight_1_diff_weight_1_grad = np.zeros(self._weight_1.shape, dtype=np.uint64)
        weight_1_diff_weight_1_grad[weight_1_mask] = (-1 * (weight_1_grad[weight_1_mask] -
                                                            self._weight_1[weight_1_mask])
                                                      .astype(np.int64)) % self.__prime
        weight_1_diff_weight_1_grad[~weight_1_mask] = self._weight_1[~weight_1_mask] - weight_1_grad[~weight_1_mask]

        self._weight_2 = weight_2_diff_weight_2_grad
        self._weight_1 = weight_1_diff_weight_1_grad

    def _forward(self, input_matrix: np.ndarray, mode: str = 'train') -> np.ndarray:
        before_activation = finite_field_truncation((input_matrix @ self._weight_1) % self.__prime,
                                                    self.__scale_input_parameter, self.__prime)
        assert before_activation.dtype == np.uint64, 'before activation is not defined in finite field domain'
        first_forward = finite_field_truncation(np.square(before_activation) % self.__prime,
                                                self.__scale_weight_parameter, self.__prime)
        assert first_forward.dtype == np.uint64, 'first forward after activation is not defined in finite field domain'
        out = finite_field_truncation((first_forward @ self._weight_2) % self.__prime, self.__scale_weight_parameter,
                                      self.__prime)
        assert out.dtype == np.uint64, 'out is not defined in finite field domain'

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

        label_mask = label < out
        label_diff_out = np.zeros(label.shape, dtype=np.uint64)
        label_diff_out[label_mask] = (-1 * (out[label_mask] - label[label_mask]).astype(np.int64)) % self.__prime
        label_diff_out[~label_mask] = label[~label_mask] - out[~label_mask]

        weight_2_grad = ((-2 % self.__prime) * finite_field_truncation((first_forward.T @ label_diff_out) %
                                                                       self.__prime, self.__scale_weight_parameter,
                                                                       self.__prime)) % self.__prime
        assert weight_2_grad.dtype == np.uint64, 'gradient for second weight matrix is not defined in finite' \
                                                 ' field domain'

        # weight_1 gradients
        middle_term = ((-2 % self.__prime) * finite_field_truncation((label_diff_out @ self._weight_2.T) % self.__prime,
                                                                     self.__scale_weight_parameter,
                                                                     self.__prime)) % self.__prime
        assert middle_term.dtype == np.uint64, 'middle term is not defined in finite field domain'
        last_term = (2 * finite_field_truncation((middle_term * before_activation) % self.__prime,
                                                 self.__scale_weight_parameter, self.__prime)) % self.__prime
        assert last_term.dtype == np.uint64, 'last term is not defined in finite field domain'
        weight_1_grad = finite_field_truncation((input_matrix.T @ last_term) % self.__prime,
                                                self.__scale_input_parameter, self.__prime)
        assert weight_1_grad.dtype == np.uint64, 'gradient for first weight matrix is not defined in finite' \
                                                 ' field domain'

        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    # noinspection DuplicatedCode
    def train(self, data_path: str, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        self.__batch_size_param = int(np.log2(self.__batch_size))
        train_data, train_label, test_data_all, test_label_all = load_all_data(self.__scale_input_parameter,
                                                                               self.__scale_weight_parameter,
                                                                               self.__prime)
        train_data, train_label, test_data_all, test_label_all = create_batch_data(train_data, train_label,
                                                                                   test_data_all, test_label_all,
                                                                                   self.__batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss), verbose=False)

                if idx == 0 or (idx + 1) % 10 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self.__prime)
                        pred_label = np.argmax(test_out, axis=1)
                        curr_acc = curr_acc + np.count_nonzero(pred_label == test_label)
                        test_total = test_total + test_data.shape[0]
                    running_acc.append(curr_acc / (test_total + 1))
                    if idx == 0 or (idx + 1) % 10 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                 running_loss[-1],
                                                                                                 running_acc[-1]),
                             verbose=False)
                    curr_loss = 0
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc
    
    # noinspection DuplicatedCode
    def train_cifar10(self, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        self.__batch_size_param = int(np.log2(self.__batch_size))
        train_data, train_label, test_data_all, test_label_all = load_all_data_cifar10(self.__scale_input_parameter, 
                                                                                       self.__scale_weight_parameter,
                                                                                       self.__prime)
        train_data, train_label, test_data_all, test_label_all = create_batch_data(train_data, train_label,
                                                                                   test_data_all, test_label_all,
                                                                                   self.__batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss), verbose=False)

                if idx == 0 or (idx + 1) % 10 == 0:
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self.__prime)
                        pred_label = np.argmax(test_out, axis=1)
                        curr_acc = curr_acc + np.count_nonzero(pred_label == test_label)
                        test_total = test_total + test_data.shape[0]
                    running_acc.append(curr_acc / (test_total + 1))
                    if idx == 0 or (idx + 1) % 10 == 0:
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                 running_loss[-1],
                                                                                                 running_acc[-1]),
                             verbose=False)
                    curr_loss = 0
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc
    
    # noinspection DuplicatedCode
    def train_vgg_cifar10(self, num_of_epochs: int, learning_rate: float, batch_size: int):
        self.__batch_size = batch_size
        self.__batch_size_param = int(np.log2(self.__batch_size))
        train_data, train_label, test_data_all, test_label_all = load_all_data_apply_vgg_cifar10(self.__scale_input_parameter, 
                                                                                                 self.__scale_weight_parameter,
                                                                                                 self.__prime)
        train_data, train_label, test_data_all, test_label_all = create_batch_data(train_data, train_label,
                                                                                   test_data_all, test_label_all,
                                                                                   self.__batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        running_curr_loss = []
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer(learning_rate)
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch, idx + 1, loss))
                running_curr_loss.append(loss)

                if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self.__prime)
                        pred_label = np.argmax(test_out, axis=1)
                        curr_acc = curr_acc + np.count_nonzero(pred_label == test_label)
                        test_total = test_total + test_data.shape[0]
                    running_acc.append(curr_acc / (test_total + 1))
                    if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                        print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        info('#############epoch: {}, avg loss: {}, acc: {}#############'.format(epoch,
                                                                                                 running_loss[-1],
                                                                                                 running_acc[-1]),
                             verbose=False)
                    curr_loss = 0
                    curr_acc = 0
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        self.__running_curr_loss = running_curr_loss

