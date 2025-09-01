"""
mlsuite/starter.py
A starter file to create dataloaders and datasets compatible with Torch
"""

import random

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class Starter:
    def __init__(
        self, X, y, batch_size=16, test_size=0.2, random_state=42, num_epochs=16
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_datasets(self):
        # Encode labels
        encoder = LabelEncoder()
        y_int = encoder.fit_transform(self.y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_int, test_size=0.2, random_state=42, stratify=y_int
        )

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train).unsqueeze(
            1
        )  # shape = (samples, channels=1, timesteps)
        X_test = torch.tensor(X_test).unsqueeze(1)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

        # Create DataLoaders
        self.train_dataset = TensorDataset(X_train, y_train)
        self.test_dataset = TensorDataset(X_test, y_test)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        self.num_classes = len(encoder.classes_)
