import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FNN(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    # hyper parameters
    batch_size = 64
    lr = 0.001
    num_epoches = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    model = FNN(28*28, 300, 100, 10)
    model.to(device)

    # loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)

    # train
    for epoch in range(num_epoches):
        for data in train_loader:
            img, label = data
            img, label = img.to(device), label.to(device)
            img = img.view(img.size(0), -1)
            out = model(img)
            loss = loss_fn(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("Epoch: %3d, loss = %4f" % (epoch+1, loss.data.item()))

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            img, label = img.to(device), label.to(device)
            img = img.view(img.size(0), -1)
            out = model(img)
            _, pred = torch.max(out.data, dim=1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    print("Accuracy on the test set %.2f%%" % (100.0 * correct / total))
