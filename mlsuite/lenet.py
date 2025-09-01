import torch
from torch import nn


class LeNet(
    nn.Module,
):
    """a fundamental model for 2d image classification"""

    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, self.num_classes)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        """forward function"""
        y = self.pool1(nn.ReLU(self.conv1(x)))
        y = self.pool2(nn.ReLU(self.conv2(y)))
        y = y.view(y.shape[0], -1)
        y = nn.ReLU(self.fc3(nn.ReLU(self.fc2(nn.ReLU(self.fc1(y))))))
        return y
