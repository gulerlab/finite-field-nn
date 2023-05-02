from simple_network import VectorizedNet, SimpleNetwork, ScaledVectorizedNet
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
    elif args.mode == 'scaled-vectorized':
        scaled_net = ScaledVectorizedNet(8, 8, 10, (2 ** 26) - 5, device='cpu')
        scaled_net.train('./data', 1, 0.001)
