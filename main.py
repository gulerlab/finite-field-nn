from simple_network import VectorizedNet, SimpleNetwork, ScaledVectorizedFiniteFieldNet, ScaledVectorizedNet, \
    ScaledVectorizedIntegerNet, Net, SimpleNetworkVGGCIFAR10, SimpleNetworkReLU
import torch
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
import torchvision.transforms as transforms
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from torch.optim import Adam
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='mode of the training')
    args = parser.parse_args()

    if args.mode == 'vectorized':
        net = VectorizedNet(device='cpu')
        net.train('./data', 1, 0.001)
        running_acc = net.running_acc
        running_loss = net.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - vectorized net')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_vectorized.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - vectorized net')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_vectorized.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'torch':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))
        # load data
        train_dataset = FashionMNIST('./data', train=True, transform=transform, target_transform=target_transform,
                                     download=True)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        test_dataset = FashionMNIST('./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        model = SimpleNetwork()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss(reduction='sum')

        running_loss = []
        running_acc = []
        curr_loss = torch.zeros(1).to('cpu')
        curr_acc = 0
        for epoch in range(1):
            for idx, (data, label) in enumerate(train_loader):
                data, label = data.to('cpu'), label.to('cpu').squeeze()
                optimizer.zero_grad()

                out = model(data)
                loss = criterion(label, out)
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
                            if pred_label == test_label:
                                curr_acc = curr_acc + 1
                            test_idx = test_idx + 1
                        running_acc.append(curr_acc / (test_idx + 1))
                        if idx == 0 or (idx + 1) % 10000 == 0:
                            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        curr_loss = torch.zeros(1).to('cpu')
                        curr_acc = 0
                        model.train()
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - torch')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_torch.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - torch')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_torch.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'torch-vgg-cifar10':
        batch_size = 256
        num_of_epochs = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))
        # load data
        train_dataset = CIFAR10('./data', train=True, transform=transform, target_transform=target_transform,
                                download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = CIFAR10('./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        vgg_backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).eval()
        vgg_backbone = torch.nn.Sequential(*(list(vgg_backbone.children())[:-1])).to(device)

        last_batch_idx = int(len(train_dataset) / batch_size)
        if len(train_dataset) % batch_size != 0:
            last_batch_idx = last_batch_idx + 1

        model = SimpleNetworkVGGCIFAR10().to(device)
        optimizer = Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        for epoch in range(num_of_epochs):
            curr_loss = torch.zeros(1).to(device)
            for idx, (data, label) in enumerate(train_loader):
                model.train()
                data = data.to(device)
                with torch.no_grad():
                    data = vgg_backbone(data).reshape(data.size(0), -1).to(device)
                label = label.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(label, out)
                loss.backward()
                optimizer.step()
                curr_loss += loss * 10
                running_curr_loss.append(loss.item() * 10)

                with torch.no_grad():
                    if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                        curr_acc = 0
                        model.eval()
                        if idx == 0:
                            running_loss.append(curr_loss.item())
                        elif (idx + 1) == last_batch_idx:
                            running_loss.append((curr_loss / ((idx + 1) % 10)).item())
                        else:
                            running_loss.append((curr_loss / 10).item())
                        test_total = 0
                        for test_data, test_label in test_loader:
                            test_data = test_data.to(device)
                            test_data = vgg_backbone(test_data).reshape(test_data.size(0), -1).to(device)
                            test_label = test_label.to(device)
                            test_out = model(test_data)
                            pred_label = torch.argmax(test_out, dim=1)
                            curr_acc = curr_acc + torch.count_nonzero(pred_label == test_label)
                            test_total = test_total + test_data.size(0)
                        running_acc.append(curr_acc / test_total)
                        if idx == 0 or (idx + 1) % 10 == 0:
                            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        curr_loss = torch.zeros(1).to(device)
        elapsed_training_time = time.time() - start_training
        torch.save({
            'running_acc': running_acc,
            'running_loss': running_loss,
            'running_curr_loss': running_curr_loss
        }, '{}.pth'.format(args.mode))
        with open('elapsed_time_torch_cifar10_vgg.txt', 'w') as fp:
            fp.write('{} seconds'.format(elapsed_training_time))
    elif args.mode == 'torch-fashion-mnist':
        batch_size = 256
        num_of_epochs = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))

        # load data
        train_dataset = FashionMNIST('./data', train=True, transform=transform, target_transform=target_transform,
                                     download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = FashionMNIST('./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        last_batch_idx = int(len(train_dataset) / batch_size)
        if len(train_dataset) % batch_size != 0:
            last_batch_idx = last_batch_idx + 1

        model = SimpleNetworkReLU().to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        for epoch in range(num_of_epochs):
            curr_loss = torch.zeros(1).to(device)
            for idx, (data, label) in enumerate(train_loader):
                model.train()
                data = data.reshape(data.size(0), -1).to(device)
                label = label.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(label, out)
                loss.backward()
                optimizer.step()
                curr_loss += loss * 10
                running_curr_loss.append(loss.item() * 10)

                with torch.no_grad():
                    if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                        curr_acc = 0
                        model.eval()
                        if idx == 0:
                            running_loss.append(curr_loss.item())
                        elif (idx + 1) == last_batch_idx:
                            running_loss.append((curr_loss / ((idx + 1) % 10)).item())
                        else:
                            running_loss.append((curr_loss / 10).item())
                        test_total = 0
                        for test_data, test_label in test_loader:
                            test_data = test_data.reshape(test_data.size(0), -1).to(device)
                            test_label = test_label.to(device)
                            test_out = model(test_data)
                            pred_label = torch.argmax(test_out, dim=1)
                            curr_acc = curr_acc + torch.count_nonzero(pred_label == test_label)
                            test_total = test_total + test_data.size(0)
                        running_acc.append(curr_acc / test_total)
                        if idx == 0 or (idx + 1) % 10 == 0:
                            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        curr_loss = torch.zeros(1).to(device)
        elapsed_training_time = time.time() - start_training
        torch.save({
            'running_acc': running_acc,
            'running_loss': running_loss,
            'running_curr_loss': running_curr_loss
        }, '{}.pth'.format(args.mode))
        with open('elapsed_time/elapsed_time_torch_fashion_mnist.txt', 'w') as fp:
            fp.write('{} seconds'.format(elapsed_training_time))
    elif args.mode == 'torch-mnist':
        batch_size = 256
        num_of_epochs = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
                                             .scatter_(0, torch.tensor(y), 1))

        # load data
        train_dataset = MNIST('./data', train=True, transform=transform, target_transform=target_transform,
                              download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = MNIST('./data', train=False, transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        last_batch_idx = int(len(train_dataset) / batch_size)
        if len(train_dataset) % batch_size != 0:
            last_batch_idx = last_batch_idx + 1

        model = SimpleNetworkReLU().to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        running_loss = []
        running_acc = []
        running_curr_loss = []
        start_training = time.time()
        for epoch in range(num_of_epochs):
            curr_loss = torch.zeros(1).to(device)
            for idx, (data, label) in enumerate(train_loader):
                model.train()
                data = data.reshape(data.size(0), -1).to(device)
                label = label.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(label, out)
                loss.backward()
                optimizer.step()
                curr_loss += loss * 10
                running_curr_loss.append(loss.item() * 10)

                with torch.no_grad():
                    if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == last_batch_idx:
                        curr_acc = 0
                        model.eval()
                        if idx == 0:
                            running_loss.append(curr_loss.item())
                        elif (idx + 1) == last_batch_idx:
                            running_loss.append((curr_loss / ((idx + 1) % 10)).item())
                        else:
                            running_loss.append((curr_loss / 10).item())
                        test_total = 0
                        for test_data, test_label in test_loader:
                            test_data = test_data.reshape(test_data.size(0), -1).to(device)
                            test_label = test_label.to(device)
                            test_out = model(test_data)
                            pred_label = torch.argmax(test_out, dim=1)
                            curr_acc = curr_acc + torch.count_nonzero(pred_label == test_label)
                            test_total = test_total + test_data.size(0)
                        running_acc.append(curr_acc / test_total)
                        if idx == 0 or (idx + 1) % 10 == 0:
                            print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
                        curr_loss = torch.zeros(1).to(device)
        elapsed_training_time = time.time() - start_training
        torch.save({
            'running_acc': running_acc,
            'running_loss': running_loss,
            'running_curr_loss': running_curr_loss
        }, '{}.pth'.format(args.mode))
        with open('elapsed_time/elapsed_time_torch_mnist.txt', 'w') as fp:
            fp.write('{} seconds'.format(elapsed_training_time))
    elif args.mode == 'scaled-vectorized-real':
        scaled_net = ScaledVectorizedNet(8, 8, device='cpu')
        scaled_net.train('./data', 1, 0.001)
        running_acc = scaled_net.running_acc
        running_loss = scaled_net.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - vectorized scaled net')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_vectorized_scaled.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - vectorized scaled net')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_vectorized_scaled.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'scaled-vectorized-int':
        scaled_net = ScaledVectorizedIntegerNet(8, 8, 10, device='cpu')
        scaled_net.train('./data', 1, 0.001)
        running_acc = scaled_net.running_acc
        running_loss = scaled_net.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - vectorized scaled integer net')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_vectorized_scaled_integer.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - vectorized scaled integer net')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_vectorized_scaled_integer.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'scaled-vectorized-ff':
        scaled_net = ScaledVectorizedFiniteFieldNet(8, 8, 10, 2 ** 26 - 5, device='cpu')
        scaled_net.train('./data', 1, 0.001)
    elif args.mode == 'net':
        net = Net(device='cpu')
        net.train('./data', 1, 0.01, 256)
        torch.save({
            'weight_1': net.weight_1,
            'weight_2': net.weight_2,
            'running_acc': net.running_acc,
            'running_loss': net.running_loss,
            'running_curr_loss': net.running_curr_loss
        }, '{}-fashion-mnist.pth'.format(args.mode))
    elif args.mode == 'net-mnist':
        net = Net(device='cpu')
        net.train_mnist('./data', 1, 0.01, 256)
        torch.save({
            'weight_1': net.weight_1,
            'weight_2': net.weight_2,
            'running_acc': net.running_acc,
            'running_loss': net.running_loss,
            'running_curr_loss': net.running_curr_loss
        }, '{}.pth'.format(args.mode))
    elif args.mode == 'net-fashion-mnist-relu':
        net = Net(device='cpu')
        net.train_relu('./data', 1, 0.02, 256)
        torch.save({
            'weight_1': net.weight_1,
            'weight_2': net.weight_2,
            'running_acc': net.running_acc,
            'running_loss': net.running_loss,
            'running_curr_loss': net.running_curr_loss
        }, '{}.pth'.format(args.mode))
    elif args.mode == 'net-cifar10':
        net = Net(device='cpu', feature_size=3072, hidden_layer_size=256)
        net.train_cifar10('./data', 5, 0.001, 128)
        running_acc = net.running_acc
        running_loss = net.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - net with batch size: 128 - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_batch_128_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - net with batch size: 128 - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_real_batch_128_cifar10.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'net-vgg-cifar10':
        net = Net(device='cpu', feature_size=25088, hidden_layer_size=128)
        net.train_vgg_cifar10('./data', 1, 0.001, 256)
        running_acc = net.running_acc
        running_loss = net.running_loss
        running_curr_loss = net.running_curr_loss
        weight_1 = net.weight_1
        weight_2 = net.weight_2

        torch.save({
            'weight_1': weight_1,
            'weight_2': weight_2,
            'running_acc': running_acc,
            'running_loss': running_loss,
            'running_curr_loss': running_curr_loss
        }, '{}.pth'.format(args.mode))
        # plt.figure()
        # plt.plot(range(len(running_loss)), running_loss)
        # plt.title('loss vs. iteration - net with batch size: 128 - VGG - CIFAR10')
        # plt.xlabel('iteration')
        # plt.ylabel('loss')
        # plt.savefig('loss_batch_128_vgg_cifar10.jpeg', dpi=300)
        # plt.show()
        # plt.figure()
        # plt.plot(range(len(running_acc)), running_acc)
        # plt.title('acc vs. iteration - net with batch size: 128 - VGG - CIFAR10')
        # plt.xlabel('iteration')
        # plt.ylabel('acc')
        # plt.savefig('acc_real_batch_128_vgg_cifar10.jpeg', dpi=300)
        # plt.show()
    elif args.mode == 'net-vgg-cifar10-v2':  # NOT WORKING
        net = Net(device='cpu', feature_size=4096, hidden_layer_size=256)
        net.train_vgg_cifar10_v2('./data', 1, 0.001, 128)
        running_acc = net.running_acc
        running_loss = net.running_loss
        running_curr_loss = net.running_curr_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - net with batch size: 128 - VGGv2 - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_batch_128_vgg_cifar10_v2.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - net with batch size: 128 - VGGv2 - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_batch_128_vgg_cifar10_v2.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_curr_loss)), running_curr_loss)
        plt.title('all loss vs. iteration - net with batch size: 128 - VGGv2 - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('all_loss_batch_128_vgg_cifar10_v2.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'net-vgg-cifar10-relu':
        net = Net(device='cpu', feature_size=25088, hidden_layer_size=128)
        net.train_vgg_cifar10_relu('./data', 1, 0.001, 256)
        running_acc = net.running_acc
        running_loss = net.running_loss
        running_curr_loss = net.running_curr_loss
        weight_1 = net.weight_1
        weight_2 = net.weight_2

        torch.save({
            'weight_1': weight_1,
            'weight_2': weight_2,
            'running_acc': running_acc,
            'running_loss': running_loss,
            'running_curr_loss': running_curr_loss
        }, '{}.pth'.format(args.mode))
