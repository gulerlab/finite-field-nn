from simple_network import VectorizedNet, SimpleNetwork, ScaledVectorizedFiniteFieldNet, ScaledVectorizedNet, \
    ScaledVectorizedIntegerNet, Net
import torch
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

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
        net.train('./data', 5, 0.01, 128)
        running_acc = net.running_acc
        running_loss = net.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - net with batch size: 128')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_batch_128.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - net with batch size: 128')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_real_batch_128.jpeg', dpi=300)
        plt.show()
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
    elif args.mode == 'net-vgg-cifar10-v2': # NOT WORKING
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

