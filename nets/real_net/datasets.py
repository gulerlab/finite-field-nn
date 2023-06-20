from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch


def real_load_all_data_mnist(load_path, flatten=True):
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
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    test_dataset = MNIST(load_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    train_data, train_label = next(iter(train_loader))
    test_data, test_label = next(iter(test_loader))
    if flatten:
        train_data, train_label, test_data, test_label = train_data.squeeze(), train_label.squeeze(),\
            test_data.squeeze(), test_label.squeeze()
    train_data, train_label, test_data, test_label = train_data.numpy(), train_label.numpy(), test_data.numpy(), \
        test_label.numpy()

    if flatten:
        # reshape data
        train_data, test_data = train_data.reshape((train_data.shape[0], -1)),\
            test_data.reshape((test_data.shape[0], -1))
    return train_data, train_label, test_data, test_label
