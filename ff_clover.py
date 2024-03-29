"""
    this code is the final version of the finite field neural network implementation
    the scaling operations for both forward and backward propagation is based on the CLOVER paper
    that is basically scaling down the parameters after the degree reaches 3*(degree of quantization)
"""

# extra utils
from abc import ABC, abstractmethod
import math
import time

# numpy
import numpy as np

## ff training
from utils_numpy import to_real_domain, to_finite_field_domain, finite_field_truncation, load_all_data, \
    from_finite_field_to_int_domain, create_batch_data, load_all_data_cifar10, load_all_data_apply_vgg_cifar10, \
    load_all_data_mnist, load_all_data_mnist_v2

## logging
from utils_numpy import info


# noinspection DuplicatedCode
class AbstractFiniteField(ABC):
    """
        this class definition is basically the abstract class to represent the workflow of the implementation
        it can be used as a reference point for different setups/architectures of a finite field neural network
    """

    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, prime,
                 feature_size=784, hidden_layer_size=128, num_classes=10):
        self._feature_size = feature_size
        self._hidden_layer_size = hidden_layer_size
        self._num_classes = num_classes

        self._scale_input_parameter = scale_input_parameter
        self._scale_weight_parameter = scale_weight_parameter
        self._scale_learning_rate_parameter = scale_learning_rate_parameter
        self._prime = prime

        self._batch_size = None
        self._batch_size_scaling_factor = None
        self._weight_1 = None
        self._weight_2 = None

        self.__init_weight()
        self._elapsed_time = -1

    def __init_weight(self):
        """
            weights are initialized using random sampling from the range [-1/sqrt(fan_in), 1/sqrt(fan_in)] (Xavier init)
        """
        range_low, range_high = -1 / math.sqrt(self._feature_size), 1 / math.sqrt(self._feature_size)
        self._weight_1 = range_low + np.random.rand(self._feature_size, self._hidden_layer_size) * (range_high
                                                                                                    - range_low)
        range_low, range_high = -1 / math.sqrt(self._hidden_layer_size), 1 / math.sqrt(self._hidden_layer_size)
        self._weight_2 = range_low + np.random.rand(self._hidden_layer_size, self._num_classes) * (range_high
                                                                                                   - range_low)
        info('weights are initialized')

    # getters and setters for protected variables
    @property
    def scale_input_parameter(self):
        return self._scale_input_parameter

    @property
    def scale_weight_parameter(self):
        return self._scale_weight_parameter

    @property
    def scale_learning_rate_parameter(self):
        return self._scale_learning_rate_parameter

    @property
    def prime(self):
        return self._prime

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def hidden_layer_size(self):
        return self._hidden_layer_size

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value

    @property
    def weight_1(self):
        return self._weight_1

    @property
    def weight_2(self):
        return self._weight_2

    @property
    def elapsed_time(self):
        return self._elapsed_time

    # basis methods that should be used in every finite field setup
    @abstractmethod
    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> float:
        """
        loss function definition
        :param label: the ground truth labels
        :param prediction: predicted labels
        :return: the resulting loss
        """
        pass

    @abstractmethod
    def _optimizer(self):
        """
        weight optimizer
        """
        pass

    @abstractmethod
    def _forward(self, data: np.ndarray, mode: str = 'train') -> np.ndarray:
        """
        forward pass of the training for one iteration
        :param data: the input data for training
        :param mode: if train then the calculated parameters are stored to calculate gradient
            if eval this calculated parameters are not saved
        :return the predictions
        """
        pass

    @abstractmethod
    def _backward(self):
        """
        backward pass of the training for one iteration
        """
        pass

    @abstractmethod
    def train(self, num_of_epochs: int, batch_size: int):
        """
        whole training for this neural network
        :param num_of_epochs: number of epochs
        :param batch_size: batch size
        """
        pass


# noinspection DuplicatedCode
class FiniteFieldClover(AbstractFiniteField):
    def __init__(self, scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, prime, **kwargs):
        super().__init__(scale_input_parameter, scale_weight_parameter, scale_learning_rate_parameter, prime, **kwargs)
        self.__save_for_backward = None
        self.__gradients = None

        self.__running_loss = None
        self.__running_acc = None
        self.__running_curr_loss = None
        # self.batch_size_param = None

        self.__scale_init_weight()

    def __scale_init_weight(self):
        self._weight_1 = to_finite_field_domain(self._weight_1, self._scale_weight_parameter, self._prime)
        self._weight_2 = to_finite_field_domain(self._weight_2, self._scale_weight_parameter, self._prime)

    @property
    def running_loss(self):
        return self.__running_loss

    @property
    def running_acc(self):
        return self.__running_acc

    @property
    def running_curr_loss(self):
        return self.__running_curr_loss

    def _criterion(self, label: np.ndarray, prediction: np.ndarray) -> float:
        num_of_samples = prediction.shape[0]
        self.__save_for_backward['label'] = label
        real_label = to_real_domain(label, self._scale_weight_parameter, self._prime)
        assert real_label.dtype == np.float64, 'ground truth labels are not in real domain in loss'
        real_prediction = to_real_domain(prediction, self._scale_weight_parameter, self._prime)
        assert real_label.dtype == np.float64, 'predictions are not in real domain in loss'
        diff = real_label - real_prediction
        diff_norm = np.linalg.norm(diff)
        return np.square(diff_norm) / num_of_samples

    def _optimizer(self):
        # scaling down with learning rate
        weight_2_grad = finite_field_truncation(self.__gradients['weight_2_grad'], self._scale_learning_rate_parameter,
                                                self._prime)
        weight_1_grad = finite_field_truncation(self.__gradients['weight_1_grad'], self._scale_learning_rate_parameter,
                                                self._prime)

        # finite field subtraction for weight 2
        weight_2_mask = self._weight_2 < weight_2_grad
        weight_2_diff_weight_2_grad = np.zeros(self._weight_2.shape, dtype=np.uint64)
        weight_2_diff_weight_2_grad[weight_2_mask] = (-1 * (weight_2_grad[weight_2_mask] -
                                                            self._weight_2[weight_2_mask])
                                                      .astype(np.int64)) % self._prime
        weight_2_diff_weight_2_grad[~weight_2_mask] = self._weight_2[~weight_2_mask] - weight_2_grad[~weight_2_mask]

        # finite field subtraction for weight 1
        weight_1_mask = self._weight_1 < weight_1_grad
        weight_1_diff_weight_1_grad = np.zeros(self._weight_1.shape, dtype=np.uint64)
        weight_1_diff_weight_1_grad[weight_1_mask] = (-1 * (weight_1_grad[weight_1_mask] -
                                                            self._weight_1[weight_1_mask])
                                                      .astype(np.int64)) % self._prime
        weight_1_diff_weight_1_grad[~weight_1_mask] = self._weight_1[~weight_1_mask] - weight_1_grad[~weight_1_mask]

        # updating the weights
        self._weight_2 = weight_2_diff_weight_2_grad
        self._weight_1 = weight_1_diff_weight_1_grad

    def _forward(self, input_matrix: np.ndarray, mode: str = 'train') -> np.ndarray:
        # truncation: s_x + s_w -> s_w
        # this is because this is the input layer - edge case
        before_activation = finite_field_truncation((input_matrix @ self._weight_1) % self._prime,
                                                    self._scale_input_parameter, self._prime)  # degree: s_w
        assert before_activation.dtype == np.uint64, 'before activation is not defined in finite field domain'

        first_forward = np.square(before_activation) % self._prime  # degree: 2 * s_w
        assert first_forward.dtype == np.uint64, 'first forward after activation is not defined in finite field domain'

        # truncation: 3 * s_w -> s_w
        out = finite_field_truncation((first_forward @ self._weight_2) % self._prime, 2 * self._scale_weight_parameter,
                                      self._prime)  # degree: s_w
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
        # degree: 2 * s_w, s_w, s_w, s_x, s_w
        first_forward, out, label, input_matrix, before_activation = self.__save_for_backward['first_forward'], \
            self.__save_for_backward['out'], self.__save_for_backward['label'], \
            self.__save_for_backward['input_matrix'], self.__save_for_backward['before_activation']

        # common error term calculation - finite field subtraction of error
        label_mask = label < out
        label_diff_out = np.zeros(label.shape, dtype=np.uint64)
        label_diff_out[label_mask] = (-1 * (out[label_mask] - label[label_mask]).astype(np.int64)) % self._prime
        label_diff_out[~label_mask] = label[~label_mask] - out[~label_mask]  # degree: s_w

        # weight_2 gradients
        # truncation: 3 * s_w -> s_w
        weight_2_grad = ((-2 % self._prime) * finite_field_truncation((first_forward.T @ label_diff_out) %
                                                                      self._prime, 2 * self._scale_weight_parameter,
                                                                      self._prime)) % self._prime  # degree: s_w
        assert weight_2_grad.dtype == np.uint64, 'gradient for second weight matrix is not defined in finite' \
                                                 ' field domain'

        # weight_1 gradients
        # degree: 2 * s_w
        middle_term = ((-2 % self._prime) * ((label_diff_out @ self._weight_2.T) % self._prime)) % self._prime
        assert middle_term.dtype == np.uint64, 'middle term is not defined in finite field domain'

        # truncation: 3 * s_w -> s_w
        # degree: s_w
        # this is the derivative of the error propagated
        last_term = (2 * finite_field_truncation((middle_term * before_activation) % self._prime,
                                                 2 * self._scale_weight_parameter, self._prime)) % self._prime
        assert last_term.dtype == np.uint64, 'last term is not defined in finite field domain'

        # truncation: s_x + s_w -> s_w
        # this is because this is the input layer - edge case
        # TODO: make it more generalizable (propagate error, multiply with the input of that layer, if 3 * s_w truncate)
        weight_1_grad = finite_field_truncation((input_matrix.T @ last_term) % self._prime,
                                                self._scale_input_parameter, self._prime)  # degree: s_w
        assert weight_1_grad.dtype == np.uint64, 'gradient for first weight matrix is not defined in finite' \
                                                 ' field domain'

        # scaling the average gradient
        # FIXME: this is not the actual implementation, just a representation of decoding and aggregating
        weight_1_grad = finite_field_truncation(weight_1_grad, self._batch_size_scaling_factor, self._prime)
        weight_2_grad = finite_field_truncation(weight_2_grad, self._batch_size_scaling_factor, self._prime)
        self.__gradients = {
            'weight_2_grad': weight_2_grad,
            'weight_1_grad': weight_1_grad
        }

    # noinspection DuplicatedCode
    def train(self, num_of_epochs: int, batch_size: int):
        """
        This training is for Fashion MNIST by default
        :param num_of_epochs: number of epochs
        :param batch_size: batch size
        """
        self._batch_size = batch_size
        self._batch_size_scaling_factor = int(np.log2(self._batch_size))

        # data loading
        train_data, train_label, test_data_all, test_label_all = load_all_data(self._scale_input_parameter,
                                                                               self._scale_weight_parameter,
                                                                               self._prime)
        train_data, train_label, test_data_all, test_label_all = create_batch_data(train_data, train_label,
                                                                                   test_data_all, test_label_all,
                                                                                   self._batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                # train
                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer()
                curr_loss += loss
                running_curr_loss.append(loss)
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss))

                # eval
                if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                    if idx == 0:
                        running_loss.append(curr_loss)
                    elif (idx + 1) == len(train_data):
                        running_loss.append(curr_loss / ((idx + 1) % 10))
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self._prime)
                        pred_label = np.argmax(test_out, axis=1)

                        # accuracy
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
        self._elapsed_time = time.time() - start_training
        info('elapsed time: {} seconds'.format(self._elapsed_time))
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        self.__running_curr_loss = running_curr_loss

    # noinspection DuplicatedCode
    def train_cifar10(self, num_of_epochs: int, batch_size: int):
        """
            CIFAR10 from scratch experiment implementation
            :param num_of_epochs: number of epochs
            :param batch_size: batch size
        """
        self._batch_size = batch_size
        self._batch_size_scaling_factor = int(np.log2(self._batch_size))

        # data loading
        train_data, train_label, test_data_all, test_label_all = load_all_data_cifar10(self._scale_input_parameter,
                                                                                       self._scale_weight_parameter,
                                                                                       self._prime)
        train_data, train_label, test_data_all, test_label_all = create_batch_data(train_data, train_label,
                                                                                   test_data_all, test_label_all,
                                                                                   self._batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                # train
                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer()
                curr_loss += loss
                running_curr_loss.append(loss)
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss))

                # eval
                if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                    if idx == 0:
                        running_loss.append(curr_loss)
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self._prime)
                        pred_label = np.argmax(test_out, axis=1)

                        # accuracy
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
        self._elapsed_time = time.time() - start_training
        info('elapsed time: {} seconds'.format(self._elapsed_time))
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        self.__running_curr_loss = running_curr_loss

    # noinspection DuplicatedCode
    def train_vgg_cifar10(self, num_of_epochs: int, batch_size: int):
        """
            CIFAR10 using VGG as feature extractor experiment implementation
            :param num_of_epochs: number of epochs
            :param batch_size: batch size
        """
        self._batch_size = batch_size
        self._batch_size_scaling_factor = int(np.log2(self._batch_size))

        # data loading
        train_data, train_label, test_data_all, test_label_all = load_all_data_apply_vgg_cifar10(
            self._scale_input_parameter,
            self._scale_weight_parameter,
            self._prime
        )
        train_data, train_label, test_data_all, test_label_all = create_batch_data(train_data, train_label,
                                                                                   test_data_all, test_label_all,
                                                                                   self._batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                # train
                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer()
                curr_loss += loss
                info('epoch: {}, iter: {}, loss: {}'.format(epoch, idx + 1, loss))
                running_curr_loss.append(loss)

                # eval
                if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                    if idx == 0:
                        running_loss.append(curr_loss)
                    elif (idx + 1) == len(train_data):
                        running_loss.append(curr_loss / ((idx + 1) % 10))
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self._prime)
                        pred_label = np.argmax(test_out, axis=1)

                        # accuracy
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
        self._elapsed_time = time.time() - start_training
        info('elapsed time: {} seconds'.format(self._elapsed_time))
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        self.__running_curr_loss = running_curr_loss

    # noinspection DuplicatedCode
    def train_mnist(self, num_of_epochs: int, batch_size: int):
        """
            MNIST experiment implementation
            :param num_of_epochs: number of epochs
            :param batch_size: batch size
        """
        self._batch_size = batch_size
        self._batch_size_scaling_factor = int(np.log2(self._batch_size))

        # data loading
        train_data, train_label, test_data_all, test_label_all = load_all_data_mnist(self._scale_input_parameter,
                                                                                     self._scale_weight_parameter,
                                                                                     self._prime)
        train_data, train_label, test_data_all, test_label_all = create_batch_data(train_data, train_label,
                                                                                   test_data_all, test_label_all,
                                                                                   self._batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        for epoch in range(num_of_epochs):
            curr_loss = 0
            curr_acc = 0
            for idx, (data, label) in enumerate(zip(train_data, train_label)):
                # train
                out = self._forward(data)
                loss = self._criterion(label, out)
                self._backward()
                self._optimizer()
                curr_loss += loss
                running_curr_loss.append(loss)
                info('epoch: {}, iter: {}, loss: {}'.format(epoch + 1, idx + 1, loss))

                # eval
                if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                    if idx == 0:
                        running_loss.append(curr_loss)
                    elif (idx + 1) == len(train_data):
                        running_loss.append(curr_loss / ((idx + 1) % 10))
                    else:
                        running_loss.append((curr_loss / 10))
                    test_total = 0
                    for test_data, test_label in zip(test_data_all, test_label_all):
                        test_out = self._forward(test_data, mode='eval')
                        test_out = from_finite_field_to_int_domain(test_out, self._prime)
                        pred_label = np.argmax(test_out, axis=1)

                        # accuracy
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
        self._elapsed_time = time.time() - start_training
        info('elapsed time: {} seconds'.format(self._elapsed_time))
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        self.__running_curr_loss = running_curr_loss
    
    # noinspection DuplicatedCode
    def train_mnist_v2(self, num_of_iterations: int, batch_size: int):
        """
            MNIST experiment implementation
            :param num_of_epochs: number of epochs
            :param batch_size: batch size
        """
        self._batch_size = batch_size
        self._batch_size_scaling_factor = int(np.log2(self._batch_size))

        # data loading
        train_data, train_label, test_data_all, test_label_all = load_all_data_mnist_v2(self._scale_input_parameter,
                                                                                     self._scale_weight_parameter,
                                                                                     self._prime,
                                                                                     num_of_iterations,
                                                                                     batch_size)
        info('datasets and loaders are initialized')

        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        curr_loss = 0
        curr_acc = 0
        for idx, (data, label) in enumerate(zip(train_data, train_label)):
            # train
            out = self._forward(data)
            loss = self._criterion(label, out)
            self._backward()
            self._optimizer()
            curr_loss += loss
            running_curr_loss.append(loss)
            info('iter: {}, loss: {}'.format(idx + 1, loss))

            # eval
            if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                if idx == 0:
                    running_loss.append(curr_loss)
                elif (idx + 1) == len(train_data):
                    running_loss.append(curr_loss / ((idx + 1) % 10))
                else:
                    running_loss.append((curr_loss / 10))
                test_total = 0
                for test_data, test_label in zip(test_data_all, test_label_all):
                    test_out = self._forward(test_data, mode='eval')
                    test_out = from_finite_field_to_int_domain(test_out, self._prime)
                    pred_label = np.argmax(test_out, axis=1)

                    # accuracy
                    curr_acc = curr_acc + np.count_nonzero(pred_label == test_label)
                    test_total = test_total + test_data.shape[0]
                running_acc.append(curr_acc / (test_total + 1))
                if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == len(train_data):
                    print('loss: {}, acc: {}'.format(running_loss[-1], running_acc[-1]))
                    info('#############avg loss: {}, acc: {}#############'.format(running_loss[-1], running_acc[-1]), verbose=False)
                curr_loss = 0
                curr_acc = 0
        self._elapsed_time = time.time() - start_training
        info('elapsed time: {} seconds'.format(self._elapsed_time))
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        self.__running_curr_loss = running_curr_loss
