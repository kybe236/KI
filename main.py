import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import datasets, transforms

kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

train_data = DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=4,
    shuffle=True,
    **kwargs
)

test_data = DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=4,
    shuffle=True,
    **kwargs
)


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.conv_drop(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Netz().to(device)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_data.dataset),
                    100.0 * batch_idx / len(train_data),
                    loss.item(),
                )
            )


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss = test_loss / len(test_data.dataset)
    accuracy = 100.0 * correct / len(test_data.dataset)
    print(f"Test set: Average loss: {loss:.4f}, Accuracy: {accuracy:.0f}% Correct: {correct}")


if os.path.isfile("model.pt"):
    inp = input("test? (y/n): ")
    model.load_state_dict(torch.load("model.pt"))
    if inp == "y":
        test()
    else:
        inp = input("train? (y/n): ")
        if inp == "y":
            inp = input("how many epochs? (int): ")
            lr = float(input("learning rate? (float): "))
            momentum = float(input("momentum? (float): "))
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            for i in range(int(inp)):
                train(int(i))
                torch.save(model.state_dict(), "model.pt")
else:
    inp = input("how many epochs? (int): ")
    lr = float(input("learning rate? (float): "))
    momentum = float(input("momentum? (float): "))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for i in range(int(inp)):
        train(int(i))
        torch.save(model.state_dict(), "model.pt")
