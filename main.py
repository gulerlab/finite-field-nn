from simple_network import VectorizedNet, SimpleNetwork, ScaledVectorizedNet
import torch
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # net = VectorizedNet(device='cpu')
    # net.train('./data', 5, 0.001)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    #
    # target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float)
    #                                      .scatter_(0, torch.tensor(y), 1))
    # # load data
    # train_dataset = FashionMNIST('./data', train=True, transform=transform, target_transform=target_transform,
    #                              download=True)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #
    # test_dataset = FashionMNIST('./data', train=False, transform=transform, download=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    #
    # model = SimpleNetwork()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # criterion = torch.nn.MSELoss(reduction='sum')
    #
    # running_loss = []
    # running_acc = []
    # curr_loss = torch.zeros(1).to('cpu')
    # curr_acc = 0
    # for epoch in range(5):
    #     for idx, (data, label) in enumerate(train_loader):
    #         data, label = data.to('cpu'), label.to('cpu').squeeze()
    #         optimizer.zero_grad()
    #
    #         out = model(data)
    #         loss = criterion(label, out)
    #         loss.backward()
    #         optimizer.step()
    #
    #         curr_loss += loss
    #
    #         with torch.no_grad():
    #             if idx == 0 or (idx + 1) % 10000 == 0:
    #                 if idx == 0:
    #                     running_loss.append(curr_loss.item())
    #                 else:
    #                     running_loss.append((curr_loss / 10000).item())
    #                 model.eval()
    #                 test_idx = 1
    #                 for test_data, test_label in test_loader:
    #                     test_data, test_label = test_data.to('cpu'), test_label.to('cpu')
    #                     test_out = model(test_data)
    #                     pred_label = torch.argmax(test_out)
    #                     if pred_label == test_label:
    #                         curr_acc = curr_acc + 1
    #                     test_idx = test_idx + 1
    #                 running_acc.append(curr_acc / (test_idx + 1))
    #                 print('epoch: {}, loss: {}, acc: {}'.format(epoch, running_loss[-1], running_acc[-1]))
    #                 curr_loss = torch.zeros(1).to('cpu')
    #                 curr_acc = 0
    #                 model.train()
    scaled_net = ScaledVectorizedNet(8, 8, 10, (2 ** 26) - 5, device='cpu')
    scaled_net.train('./data', 5, 0.001)
