import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # (3, 32, 32) -> (64, 32, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # (64, 32, 32) -> (64, 16, 16)
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (64, 16, 16) -> (128, 16, 16)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # (128, 16, 16) -> (128, 8, 8)
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (128, 8, 8) -> (256, 8, 8)
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # (256, 8, 8) -> (256, 4, 4)
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # (256, 4, 4) -> (10)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # hyper parameters
    batch_size = 64
    lr = 0.001
    num_epoches = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.CIFAR10(root='./CIFAR', train=True, transform=data_tf, download=True)
    test_dataset = datasets.CIFAR10(root='./CIFAR', train=False, transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    model = CNN()
    model.to(device)

    # loss and optimizer
    loss_cnn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8)

    # train
    for epoch in range(num_epoches):
        for data in train_loader:
            img, label = data
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = loss_cnn(out, label)
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
            out = model(img)
            _, pred = torch.max(out.data, dim=1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    print("Accuracy on the test set %.2f%%" % (100.0 * correct / total))
