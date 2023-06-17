# extra utils
import numpy as np

# torch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# this project
from hist.prev_version.utils import from_int_to_real_domain, to_int_domain, to_int_domain_int, finite_field_truncation, ToIntDomain
from hist.prev_version.simple_network import AbstractVectorizedNet

import matplotlib.pyplot as plt


class GaussianRandomDataset(Dataset):
    def __init__(self, feature_size, transform=None, target_transform=None, num_of_samples=100):
        super(GaussianRandomDataset, self).__init__()
        self.__num_of_samples = num_of_samples

        self.__data = torch.zeros((num_of_samples, feature_size))
        self.__data[0:int(num_of_samples / 2), :] = torch.normal(-1, 0.5, size=(int(num_of_samples / 2), feature_size))
        self.__data[int(num_of_samples / 2):num_of_samples, :] = torch.normal(1, 0.5,
                                                                              size=(int(num_of_samples / 2),
                                                                                    feature_size))
        label_indicator = self.__data[:, 0] > 0
        self.__label = torch.zeros(num_of_samples)
        self.__label[label_indicator] = 1
        self.__label = self.__label.numpy().astype(int)

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

    def plot_data(self):
        plt.figure()
        color_arr = np.zeros(self.__data.size(0), dtype=str)
        color_map = self.__label == 0
        color_arr[color_map] = 'green'
        color_arr[np.invert(color_map)] = 'red'
        plt.scatter(self.__data[:, 0], self.__data[:, 1], c=color_arr)
        plt.show()


# noinspection DuplicatedCode
class TrackableIntNet(AbstractVectorizedNet):
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
            ToIntDomain(self.__scale_input_parameter)
        ])

        target_transform = transforms.Compose([
            transforms.Lambda(lambda y: torch.zeros(2, dtype=torch.float)
                              .scatter_(0, torch.tensor(y), 1)),
            ToIntDomain(self.__scale_weight_parameter)
        ])

        # load data
        train_dataset = GaussianRandomDataset(self.input_vector_size, transform=transform,
                                              target_transform=target_transform, num_of_samples=1000)

        # train_dataset.plot_data()
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
                        test_data = test_data.T
                        test_out = self._forward(test_data, mode='eval')
                        pred_label = torch.argmax(test_out)
                        test_label = torch.argmax(test_label)
                        if pred_label == test_label:
                            curr_acc = curr_acc + 1
                        test_idx = test_idx + 1
                    running_acc.append(curr_acc / (test_idx + 1))
                    # if idx == 0 or (idx + 1) % 100 == 0:
                    #     print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                    curr_loss = torch.zeros(1).to(self.device)
                    curr_acc = 0
            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
        self.__running_loss = running_loss
        self.__running_acc = running_acc
        # torch.save({
        #     'model': {
        #         'weight_1': self._weight_1,
        #         'weight_2': self._weight_2
        #     },
        #     'min_weight_1': self.__min_weight_value_1,
        #     'max_weight_1': self.__max_weight_value_1,
        #     'min_weight_2': self.__min_weight_value_2,
        #     'max_weight_2': self.__max_weight_value_2,
        #     'min_input': self.__min_input_value,
        #     'max_input': self.__max_input_value,
        #     'running_loss': self.__running_loss,
        #     'running_acc': self.__running_loss
        # }, 'params/trackable_int_nn_params.tar.gz')


class SimpleNetwork(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.hidden_layer = nn.Linear(2, 2, bias=False)
        self.output_layer = nn.Linear(2, num_class, bias=False)

    def forward(self, data):
        data = self.hidden_layer(data)
        data = torch.square(data)
        data = self.output_layer(data)
        return data


net = SimpleNetwork()

target_transform = transforms.Lambda(lambda y: torch.zeros(2, dtype=torch.float)
                                     .scatter_(0, torch.tensor(y), 1))
# load data
train_dataset = GaussianRandomDataset(2, target_transform=target_transform, num_of_samples=1000)
train_dataset.plot_data()

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

model = SimpleNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00000000001)
criterion = torch.nn.MSELoss(reduction='sum')

running_loss = []
running_acc = []
curr_loss = torch.zeros(1).to('cpu')
curr_acc = 0
for epoch in range(10):
    for idx, (data, label) in enumerate(train_loader):
        data, label = data.to('cpu'), label.to('cpu').squeeze()
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(label, out.reshape(-1))
        loss.backward()
        optimizer.step()

        curr_loss += loss

        with torch.no_grad():
            if idx == 0 or (idx + 1) % 100 == 0:
                if idx == 0:
                    running_loss.append(curr_loss.item())
                else:
                    running_loss.append((curr_loss / 100).item())
                model.eval()
                test_idx = 1
                for test_data, test_label in test_loader:
                    test_data, test_label = test_data.to('cpu'), test_label.to('cpu')
                    test_out = model(test_data)
                    pred_label = torch.argmax(test_out)
                    test_label = torch.argmax(test_label)
                    if pred_label == test_label:
                        curr_acc = curr_acc + 1
                    test_idx = test_idx + 1
                running_acc.append(curr_acc / (test_idx + 1))
                if idx == 0 or (idx + 1) % 100 == 0:
                    print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                curr_loss = torch.zeros(1).to('cpu')
                curr_acc = 0
                model.train()

# trackable_net = TrackableIntNet(8, 8, 17, input_vector_size=2, num_classes=2, hidden_layer_size=2, device='cpu')
# trackable_net.train('', 10, 0.00001)
