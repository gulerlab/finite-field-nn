from utils import to_finite_field_domain, info

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.models import vgg16_bn, VGG16_BN_Weights


#############
# numpy data
#############

# noinspection DuplicatedCode
def load_all_data_fashion_mnist(load_path, quantization_bit_data: int, quantization_bit_label: int, prime: int, field,
                                flatten=True):
    # transformations
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = FashionMNIST(load_path, train=True, transform=transform, target_transform=target_transform,
                                 download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = FashionMNIST(load_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = next(iter(train_loader))
    test_data, test_label = next(iter(test_loader))
    if flatten:
        train_data, train_label, test_data, test_label = train_data.squeeze(), train_label.squeeze(), \
            test_data.squeeze(), test_label.squeeze()
    train_data, train_label, test_data, test_label = train_data.numpy(), train_label.numpy(), test_data.numpy(), \
        test_label.numpy()
    train_data, train_label, test_data = to_finite_field_domain(train_data, quantization_bit_data, prime), \
        to_finite_field_domain(train_label, quantization_bit_label, prime), \
        to_finite_field_domain(test_data, quantization_bit_data, prime)
    train_data, train_label, test_data = field(train_data), field(train_label), field(test_data)
    if flatten:
        # reshape data
        train_data, test_data = train_data.reshape((train_data.shape[0], -1)),\
            test_data.reshape((test_data.shape[0], -1))
    return train_data, train_label, test_data, test_label


# noinspection DuplicatedCode
def load_all_data_mnist(load_path, quantization_bit_data: int, quantization_bit_label: int, prime: int, field,
                        flatten=True):
    # transformations
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = MNIST(load_path, train=True, transform=transform, target_transform=target_transform,
                          download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = MNIST(load_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = next(iter(train_loader))
    test_data, test_label = next(iter(test_loader))
    if flatten:
        train_data, train_label, test_data, test_label = train_data.squeeze(), train_label.squeeze(),\
            test_data.squeeze(), test_label.squeeze()
    train_data, train_label, test_data, test_label = train_data.numpy(), train_label.numpy(), test_data.numpy(), \
        test_label.numpy()
    train_data, train_label, test_data = to_finite_field_domain(train_data, quantization_bit_data, prime), \
        to_finite_field_domain(train_label, quantization_bit_label, prime), \
        to_finite_field_domain(test_data, quantization_bit_data, prime)
    train_data, train_label, test_data = field(train_data), field(train_label), field(test_data)
    if flatten:
        # reshape data
        train_data, test_data = train_data.reshape((train_data.shape[0], -1)),\
            test_data.reshape((test_data.shape[0], -1))
    return train_data, train_label, test_data, test_label


def load_all_data_cifar10(load_path, quantization_bit_data: int, quantization_bit_label: int, prime: int, field,
                          flatten=True):
    # transformations
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    target_transform = Compose([
        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))
    ])

    # load data
    train_dataset = CIFAR10(load_path, train=True, transform=transform, target_transform=target_transform,
                            download=True)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = CIFAR10(load_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_data, train_label = next(iter(train_loader))
    test_data, test_label = next(iter(test_loader))
    train_data, train_label, test_data, test_label = train_data.numpy(), train_label.numpy(), test_data.numpy(), \
        test_label.numpy()
    train_data, train_label, test_data = to_finite_field_domain(train_data, quantization_bit_data, prime), \
        to_finite_field_domain(train_label, quantization_bit_label, prime), \
        to_finite_field_domain(test_data, quantization_bit_data, prime)
    train_data, train_label, test_data = field(train_data), field(train_label), field(test_data)
    if flatten:
        # reshape data
        train_data, test_data = train_data.reshape((train_data.shape[0], -1)),\
            test_data.reshape((test_data.shape[0], -1))
    return train_data, train_label, test_data, test_label


