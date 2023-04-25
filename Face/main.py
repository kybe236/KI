import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import random

PATH = "data"
img_list = os.listdir(PATH)

#  PATH_TEST = "data_test"
#  img_list_test = os.listdir(PATH_TEST)

PATH_WRONG = "data_wrong"
img_list_wrong = os.listdir(PATH_WRONG)

device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=12, kernel_size=5
        )
        self.fc1 = nn.Linear(41772, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        print(x.shape)
        print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)


net = Net().to(device)

losses = []


def train(epochs):
    net.train()
    t = 0
    print("Training")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(epochs):
        torch.save(net.state_dict(), "model.pt")
        random.shuffle(img_list)
        random.shuffle(img_list_wrong)

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
            if t % 25 == 0:
                print(f"loss: {running_loss / 25}")
                running_loss = 0.0

        print(f"Epoch: {epoch + 1}, loss: {epoch_loss / len(img_list)}")


# def test():
#     print("Testing")
#     correct = 0
#    total = 0
#    with torch.no_grad():
#        for i in img_list_test:
#            img = Image.open(f"{PATH_TEST}/{i}")
#            img = img.resize((500, 500))
#            img = transforms.ToTensor()(img)
#            img = img.unsqueeze(0)
#
#            label = 1
#
#            outputs = net(img)
#            _, predicted = torch.max(outputs.data, 1)
#
#            total += len(img_list_test)
#            correct += (predicted == label).sum().item()
#    print(f"Accuracy: {100 * correct / total}")



def main():
    if os.path.isfile("model.pt"):
        inp = input("Do you want to load the model? (y/n) ")

        if inp == "y":
            net.load_state_dict(torch.load("model.pt"))
            print("Model loaded")

        # inp = input("Do you want to test the model? (y/n) ")
        # if inp == "y":
        #    test()
        #    sys.exit()

    inp = input("Do you want to train the model? (y/n) ")
    if inp == "y":
        epochs = int(input("How many epochs? "))
        train(epochs)
        sys.exit()


if __name__ == "__main__":
    main()
