"""
mlsuite/cnn1d.py
A basic CNN for multi-class classification of IFFT RF data
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """a simple one-dimensional CNN"""

    def __init__(self, input_length, num_classes):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        # Compute the size after conv/pool layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy = self.pool1(torch.relu(self.conv1(dummy)))
            dummy = self.pool2(torch.relu(self.conv2(dummy)))
            dummy = self.pool3(torch.relu(self.conv3(dummy)))
            flattened_size = dummy.numel()

        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """forward function for cnn"""
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
