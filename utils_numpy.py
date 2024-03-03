import logging
import numpy as np
from numpy import ndarray
import math

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.models import vgg16_bn, VGG16_BN_Weights


############
# domain converting operations
############


def to_finite_field_domain(real: ndarray, quantization_bit: int, prime: int) -> ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real)
    finite_field = np.zeros(int_domain.shape, dtype=np.uint64)
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = int_domain[~negative_mask]
    finite_field[negative_mask] = prime - (int_domain[negative_mask] * -1).astype(np.uint64)
    return finite_field


def to_int_domain(real: ndarray, quantization_bit: int) -> ndarray:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = np.around(scaled_real).astype(np.int64)
    return int_domain


def to_real_domain(finite_field: ndarray, quantization_bit: int, prime: int) -> ndarray:
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    real_domain = np.zeros(finite_field.shape, dtype=np.float64)
    real_domain[~negative_mask] = finite_field[~negative_mask]
    real_domain[negative_mask] = -1 * (prime - finite_field[negative_mask]).astype(np.float64)
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def from_int_to_real_domain(int_domain: ndarray, quantization_bit: int) -> ndarray:
    real_domain = int_domain.astype(np.float64)
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def int_truncation(int_domain: ndarray, scale_down: int) -> ndarray:
    real_domain = int_domain.astype(np.int64)
    real_domain = real_domain / (2 ** scale_down)
    real_domain_floor = np.floor(real_domain)

    zero_distributions = real_domain - real_domain_floor
    stochastic_fnc = np.vectorize(lambda x: np.random.choice([0, 1], 1, p=[1 - x, x])[0])
    zero_distributions = stochastic_fnc(zero_distributions)

    truncated_int_domain = (real_domain_floor + zero_distributions).astype(np.int64)
    return truncated_int_domain


def from_finite_field_to_int_domain(finite_field: ndarray, prime: int) -> ndarray:
    int_domain = np.zeros(finite_field.shape, dtype=np.int64)
    threshold = (prime - 1) / 2
    negative_mask = finite_field > threshold
    int_domain[~negative_mask] = finite_field[~negative_mask]
    int_domain[negative_mask] = -1 * (prime - finite_field[negative_mask]).astype(np.int64)
    return int_domain


def from_int_to_finite_field_domain(int_domain: ndarray, prime: int) -> ndarray:
    finite_field = np.zeros(int_domain.shape, dtype=np.uint64)
    negative_mask = int_domain < 0
    finite_field[~negative_mask] = int_domain[~negative_mask]
    finite_field[negative_mask] = int_domain[negative_mask] + prime
    return finite_field


def finite_field_truncation(finite_field: ndarray, scale_down: int, prime: int) -> ndarray:
    int_domain = from_finite_field_to_int_domain(finite_field, prime)
    int_domain = int_truncation(int_domain, scale_down)
    finite_field_domain = from_int_to_finite_field_domain(int_domain, prime)
    return finite_field_domain


# noinspection DuplicatedCode
def to_finite_field_domain_int(real: float, quantization_bit: int, prime: int) -> int:
    scaled_real = real * (2 ** quantization_bit)
    finite_field_domain = round(scaled_real)
    if finite_field_domain < 0:
        finite_field_domain = finite_field_domain + prime
    return int(finite_field_domain)


def to_int_domain_int(real: float, quantization_bit: int) -> int:
    scaled_real = real * (2 ** quantization_bit)
    int_domain = round(scaled_real)
    return int(int_domain)


def to_real_domain_int(finite_field: int, quantization_bit: int, prime: int) -> ndarray:
    threshold = (prime - 1) / 2
    real_domain = finite_field
    if real_domain > threshold:
        real_domain = real_domain - prime
    real_domain = real_domain / (2 ** quantization_bit)
    return real_domain


def from_int_to_real_domain_int(int_domain: int, quantization_bit: int):
    real_domain = int_domain / (2 ** quantization_bit)
    return real_domain


def finite_field_truncation_int(finite_field: int, scale_down: int) -> int:
    real_domain = finite_field / (2 ** scale_down)
    real_domain_floor = math.floor(real_domain)
    remainder = real_domain - real_domain_floor
    random_bit = np.random.choice([0, 1], 1, p=[1 - remainder, remainder])[0]
    finite_field_domain = int(real_domain_floor + random_bit)
    return finite_field_domain


###################
# transformers
###################


class ToFiniteFieldDomain(object):
    def __init__(self, scale_input_parameter, prime):
        self.__scale_input_parameter = scale_input_parameter
        self.__prime = prime

    @property
    def scale_input_parameter(self):
        return self.__scale_input_parameter

    @scale_input_parameter.setter
    def scale_input_parameter(self, value):
        self.__scale_input_parameter = value

    @property
    def prime(self):
        return self.__prime

    @prime.setter
    def prime(self, value):
        self.__prime = value

    def __call__(self, sample):
        return to_finite_field_domain(sample, self.__scale_input_parameter, self.__prime)


class ToIntDomain(object):
    def __init__(self, scale_input_parameter):
        self.__scale_input_parameter = scale_input_parameter

    @property
    def scale_input_parameter(self):
        return self.__scale_input_parameter

    @scale_input_parameter.setter
    def scale_input_parameter(self, value):
        self.__scale_input_parameter = value

    def __call__(self, sample):
        return to_int_domain(sample, self.__scale_input_parameter)


class ToNumpy(object):
    def __call__(self, sample):
        return sample.numpy()


#############
# numpy data
#############

def collect_augment_aggregate_data(dataset, dataloader, num_of_clients, num_of_classes):
    data, labels = next(iter(dataloader))
    data, labels = data.numpy(), labels.numpy()
    targets = dataset.targets
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)

    different_classes_data = []
    different_classes_labels = []
    for class_id in range(num_of_classes):
        different_classes_data.append(np.array_split(data[targets == class_id], num_of_clients))
        different_classes_labels.append(np.array_split(labels[targets == class_id], num_of_clients))
    
    client_data = []
    client_labels = []
    for client_idx in range(num_of_clients):
        client_data_buffer = []
        client_labels_buffer = []
        for class_idx in range(num_of_classes):
            client_data_buffer.append(different_classes_data[class_idx][client_idx])
            client_labels_buffer.append(different_classes_labels[class_idx][client_idx])

        client_data_buffer = np.concatenate(client_data_buffer)
        for channel_idx in range(client_data_buffer.shape[1]):
            client_data_buffer[:, channel_idx, :, :] = (client_data_buffer[:, channel_idx, :, :] - np.mean(client_data_buffer[:, channel_idx, :, :])) / np.std(client_data_buffer[:, channel_idx, :, :])

        client_data.append(client_data_buffer)
        client_labels.append(np.concatenate(client_labels_buffer))
    aggregated_data = np.concatenate(client_data)
    aggregated_labels = np.concatenate(client_labels)
    randomize = np.random.permutation(aggregated_data.shape[0])
    return aggregated_data[randomize], aggregated_labels[randomize]

# noinspection DuplicatedCode
def load_all_data(quantization_bit_data: int, quantization_bit_label: int, prime: int):
    # transformations
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = FashionMNIST('./data', train=True, transform=transform, target_transform=target_transform,
                                 download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = FashionMNIST('./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = next(iter(train_loader))
    test_data, test_label = next(iter(test_loader))
    train_data, train_label, test_data, test_label = train_data.squeeze(), train_label.squeeze(), test_data.squeeze(), \
        test_label.squeeze()
    train_data, train_label, test_data, test_label = train_data.numpy(), train_label.numpy(), test_data.numpy(), \
        test_label.numpy()
    train_data, train_label, test_data = to_finite_field_domain(train_data, quantization_bit_data, prime), \
        to_finite_field_domain(train_label, quantization_bit_label, prime), \
        to_finite_field_domain(test_data, quantization_bit_data, prime)
    # reshape data
    train_data, test_data = train_data.reshape((train_data.shape[0], -1)), test_data.reshape((test_data.shape[0], -1))
    return train_data, train_label, test_data, test_label


# noinspection DuplicatedCode
def load_all_data_mnist(quantization_bit_data: int, quantization_bit_label: int, prime: int, num_of_clients: int = 10):
    # transformations
    transform = Compose([
        ToTensor()
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = MNIST('./data', train=True, transform=transform, target_transform=target_transform,
                          download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = MNIST('./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = collect_augment_aggregate_data(train_dataset, train_loader, num_of_clients, 10)
    test_data, test_label = collect_augment_aggregate_data(test_dataset, test_loader, num_of_clients, 10)
    train_data, train_label, test_data = to_finite_field_domain(train_data, quantization_bit_data, prime), \
        to_finite_field_domain(train_label, quantization_bit_label, prime), \
        to_finite_field_domain(test_data, quantization_bit_data, prime)
    # reshape data
    train_data, test_data = train_data.reshape((train_data.shape[0], -1)), test_data.reshape((test_data.shape[0], -1))
    return train_data, train_label, test_data, test_label


def load_all_data_cifar10(quantization_bit_data: int, quantization_bit_label: int, prime: int):
    # transformations
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = CIFAR10('./data', train=True, transform=transform, target_transform=target_transform,
                            download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = CIFAR10('./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = next(iter(train_loader))
    test_data, test_label = next(iter(test_loader))
    train_data, train_label, test_data, test_label = train_data.numpy(), train_label.numpy(), test_data.numpy(), \
        test_label.numpy()
    train_data, train_label, test_data = to_finite_field_domain(train_data, quantization_bit_data, prime), \
        to_finite_field_domain(train_label, quantization_bit_label, prime), \
        to_finite_field_domain(test_data, quantization_bit_data, prime)
    # reshape data
    train_data, test_data = train_data.reshape((train_data.shape[0], -1)), test_data.reshape((test_data.shape[0], -1))
    return train_data, train_label, test_data, test_label


def load_all_data_apply_vgg_cifar10(quantization_bit_data: int, quantization_bit_label: int, prime: int, num_of_clients: int = 10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = Compose([
        ToTensor(),
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = CIFAR10('./data', train=True, transform=transform, target_transform=target_transform,
                            download=True)
    train_loader = DataLoader(train_dataset, batch_size=train_dataset.data.shape[0], shuffle=False)
    test_dataset = CIFAR10('./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.data.shape[0], shuffle=False)


    all_train_data, all_train_labels = collect_augment_aggregate_data(train_dataset, train_loader, num_of_clients, 10)
    all_test_data, all_test_labels = collect_augment_aggregate_data(test_dataset, test_loader, num_of_clients, 10)
    all_train_data, all_train_labels, all_test_data, all_test_labels = create_batch_data(all_train_data, all_train_labels, all_test_data, all_test_labels, 256)

    vgg_backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).eval()
    vgg_backbone = torch.nn.Sequential(*(list(vgg_backbone.children())[:-1])).to(device)


    with torch.no_grad():
        train_data_all, train_label_all, test_data_all, test_label_all = [], [], [], []
        for train_data, train_label in zip(all_train_data, all_train_labels):
            train_data = torch.tensor(train_data)
            train_data = train_data.to(device)
            train_data = vgg_backbone(train_data).reshape(train_data.size(0), -1).to('cpu').numpy()

            train_data_all.append(train_data)
            train_label_all.append(train_label)
        info('train data is handled')

        for test_data, test_label in zip(all_test_data, all_test_labels):
            test_label_all.append(test_label)
            test_data = torch.tensor(test_data)
            test_data = test_data.to(device)
            test_data = vgg_backbone(test_data).reshape(test_data.size(0), -1).to('cpu').numpy()
            test_data_all.append(test_data)
        info('test data is handled')



    train_data_all, train_label_all = np.concatenate(train_data_all, axis=0), np.concatenate(train_label_all, axis=0)
    test_data_all, test_label_all = np.concatenate(test_data_all, axis=0), np.concatenate(test_label_all, axis=0)
    train_data_all, train_label_all = to_finite_field_domain(train_data_all, quantization_bit_data, prime), \
                to_finite_field_domain(train_label_all, quantization_bit_label, prime)
    test_data_all = to_finite_field_domain(test_data_all, quantization_bit_data, prime)
    return train_data_all, train_label_all, test_data_all, test_label_all


def create_batch_data(train_data, train_label, test_data, test_label, batch_size):
    train_num_samples, test_num_samples = train_data.shape[0], test_data.shape[0]
    number_of_full_batch_train = int(train_num_samples / batch_size)
    last_batch_size_train = train_num_samples % batch_size

    number_of_full_batch_test = int(test_num_samples / batch_size)
    last_batch_size_test = test_num_samples % batch_size

    last_batch_train_data = None
    if last_batch_size_train != 0:
        last_batch_train_data = train_data[train_num_samples - last_batch_size_train:, :]

    train_data = np.split(train_data[:train_num_samples - last_batch_size_train, :], number_of_full_batch_train)
    if last_batch_train_data is not None:
        train_data.append(last_batch_train_data)

    last_batch_train_label = None
    if last_batch_size_train != 0:
        last_batch_train_label = train_label[train_num_samples - last_batch_size_train:, :]

    train_label = np.split(train_label[:train_num_samples - last_batch_size_train, :], number_of_full_batch_train)
    if last_batch_train_label is not None:
        train_label.append(last_batch_train_label)

    last_batch_test_data = None
    if last_batch_size_test != 0:
        last_batch_test_data = test_data[test_num_samples - last_batch_size_test:, :]

    test_data = np.split(test_data[:test_num_samples - last_batch_size_test, :], number_of_full_batch_test)
    if last_batch_test_data is not None:
        test_data.append(last_batch_test_data)

    last_batch_test_label = None
    if last_batch_size_test != 0:
        last_batch_test_label = test_label[test_num_samples - last_batch_size_test:]

    test_label = np.split(test_label[:test_num_samples - last_batch_size_test], number_of_full_batch_test)
    if last_batch_test_label is not None:
        test_label.append(last_batch_test_label)

    return train_data, train_label, test_data, test_label


#############
# utils for debug
#############

def info(msg, verbose=True):
    logging.info(msg)
    if verbose:
        print(msg)
