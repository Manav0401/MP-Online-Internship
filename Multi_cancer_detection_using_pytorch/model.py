import torch.nn as nn


class CancerCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32*56*56,128)
        self.fc2 = nn.Linear(128,26)

    def forward(self,x):

        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))

        x=self.flatten(x)

        x=self.relu(self.fc1(x))
        x=self.fc2(x)

        return x