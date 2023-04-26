import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import random
from matplotlib import pyplot as plt

PATH = "data"
img_list = os.listdir(PATH)

PATH_TEST = "data_test"
img_list_test = os.listdir(PATH_TEST)

PATH_WRONG = "data_wrong"
img_list_wrong = os.listdir(PATH_WRONG)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=12, kernel_size=5
        )
        self.conv3 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5
        )
        self.fc1 = nn.Linear(17496, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.dropout(x, p=0.5)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.dropout(x, p=0.3)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.dropout(x, p=0.2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


net = Net().to(device)

losses = []


def train(epochs):
    net.train()
    t = 0
    print("Training")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        torch.save(net.state_dict(), "model.pt")
        random.shuffle(img_list)
        random.shuffle(img_list_wrong)

        right = 0
        right_2 = [0]
        false = 0
        false_2 = [0]
        epoch_loss = 0.0
        running_loss = 0.0
        for i in img_list:
            if t == 100:
                torch.save(net.state_dict(), "model.pt")
            t = t + 1
            ran = random.randrange(0, 2)
            if ran == 1:
                img = Image.open(f"{PATH}/{i}").resize((250, 250)).convert("L")
                img = transforms.ToTensor()(img)
                img = img.unsqueeze(0).to(device)
                img = img[:3].to(device)
            if ran == 0:
                rand_img = random.choice(img_list_wrong)
                img = Image.open(f"{PATH_WRONG}/{rand_img}").resize((250, 250)).convert("L")
                img = transforms.ToTensor()(img)
                img = img.unsqueeze(0).to(device)
                img = img[:3]
            label = torch.tensor([ran]).to(device)
            label = label.unsqueeze(1)
            label = label.float()

            optimizer.zero_grad()

            outputs = net(img)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            if (outputs.item() > 0.5) and (label.item() == 1):
                right = right + 1
            elif (outputs.item() < 0.5) and (label.item() == 0):
                right = right + 1
            else:
                false = false + 1
            if t % 100 == 0:
                print(f"epoch: {epoch} loss: {running_loss / 100} right: {right} false: {false}")
                right_2.append(right)
                false_2.append(false)
                right = 0
                false = 0
                running_loss = 0.0
            if t % 1000 == 0:
                plt.plot(right_2, label="right")
                plt.xlim(0, 1000)
                plt.autoscale(enable=True, axis="x", tight=True)
                plt.show()
                plt.savefig("loss.png")

        print(f"Epoch: {epoch + 1}, loss: {epoch_loss / len(img_list)}")


def test():
    net.eval()
    right = 0
    false = 0
    for i in img_list_test:
        img = Image.open(f"{PATH_TEST}/{i}").resize((250, 250)).convert("L")
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)
        img = img[:3].to(device)

        label = torch.tensor([1]).to(device)
        label = label.unsqueeze(1)
        label = label.float()

        outputs = net(img)

        if (outputs.item() > 0.5) and (label.item() == 1):
            right = right + 1
        elif (outputs.item() < 0.5) and (label.item() == 0):
            right = right + 1
        else:
            false = false + 1

    print(f"right: {right} false: {false} accuracy: {right / 1000}")


def test_file():
    net.eval()

    inp = input("File name: ")
    img = Image.open(f"{inp}").resize((250, 250)).convert("L")
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device)
    img = img[:3].to(device)

    outputs = net(img)

    print(outputs.item())


def main():
    if os.path.isfile("model.pt"):

        inp = input("Do you want to load the model? (y/n) ")
        if inp == "y":
            net.load_state_dict(torch.load("model.pt"))
            print("Model loaded")

            inp = input("Do you want to test the model? (y/n) ")
            if inp == "y":

                inp = input("Test file? (y/n) ")
                if inp == "y":
                    test_file()
                    sys.exit(0)
                if inp == "n":
                    test()
                    sys.exit(0)

    inp = input("Do you want to train the model? (y/n) ")
    if inp == "y":

        epochs = int(input("How many epochs? "))
        train(epochs)
        sys.exit(0)


if __name__ == "__main__":
    main()
