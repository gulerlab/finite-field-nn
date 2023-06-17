# extra utils
import numpy as np

# torch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# this project
from hist.prev_version.utils import to_real_domain, to_finite_field_domain, ToFiniteFieldDomain, to_finite_field_domain_int, \
    finite_field_truncation_ext
from hist.prev_version.simple_network import AbstractVectorizedNet


class GaussianRandomDataset(Dataset):
    def __init__(self, feature_size, label_range, transform=None, target_transform=None, num_of_samples=100):
        super(GaussianRandomDataset, self).__init__()
        self.__num_of_samples = num_of_samples

        self.__data = torch.normal(0, 1, size=(num_of_samples, feature_size))
        self.__label = np.random.choice(label_range, num_of_samples)

        self.__transform = transform
        self.__target_transform = target_transform

    @property
    def data(self):
        return self.__data

    @property
    def label(self):
        return self.__label

    def __len__(self):
        return self.__num_of_samples

    def __getitem__(self, idx):
        return_data = self.__data[idx]
        return_label = self.__label[idx]
        if self.__transform:
            return_data = self.__transform(return_data)

        if self.__target_transform:
            return_label = self.__target_transform(return_label)

        return return_data, return_label


# noinspection DuplicatedCode
class TrackableFiniteFieldNet(AbstractVectorizedNet):
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
            ToFiniteFieldDomain(self.__scale_input_parameter, self.__prime)
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(2, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1)),
            ToFiniteFieldDomain(self.__scale_weight_parameter, self.__prime)
        ])

        # load data
        train_dataset = GaussianRandomDataset(self.input_vector_size, self.num_classes, transform=transform,
                                              target_transform=target_transform, num_of_samples=1000)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        running_loss = []
        running_acc = []
        curr_loss = torch.zeros(1).to(self.device)
        curr_acc = 0
        for epoch in range(num_of_epochs):
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.device), label.to(self.device)
                data, label = data.T, label.T

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
                        test_data = test_data.T
                        test_out = self._forward(test_data, mode='eval')
                        test_out = to_real_domain(test_out, self.__scale_weight_parameter, self.__prime)
                        pred_label = torch.argmax(test_out)
                        test_label = torch.argmax(test_label)
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


trackable_net = TrackableFiniteFieldNet(15, 15, 10, 2 ** 26 - 5,
                                        input_vector_size=2, num_classes=2, hidden_layer_size=2, device='cpu')
trackable_net.train('', 100, 0.001)
