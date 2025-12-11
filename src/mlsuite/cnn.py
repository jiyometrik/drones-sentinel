"""
src.mlsuite.cnn
All the CNNs used for classification tasks
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class SimplePSDClassifier(pl.LightningModule):
    """A simple 1D CNN for classifying PSDs, using only Conv1d and Linear layers"""

    def __init__(self, num_classes, lr=1e-3, weight_decay=1e-4):
        super(SimplePSDClassifier, self).__init__()
        self.save_hyperparameters()
        self.conv1_3 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(2, 32, kernel_size=5, padding=2)
        self.conv1_7 = nn.Conv1d(2, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(96)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(96, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(192)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # -> (192, 64)

        self.conv3 = nn.Conv1d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(384)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # global avg pooling
        self.gap = nn.AdaptiveAvgPool1d(1)  # -> (256, 1)

        # fully connected layers
        self.fc1 = nn.Linear(384, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, num_classes)

        # some other attributes
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc, self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ), torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """forward pass through the network"""
        x1, x2, x3 = (
            F.relu(self.conv1_3(x)),
            F.relu(self.conv1_5(x)),
            F.relu(self.conv1_7(x)),
        )
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pool1(self.bn1(x))

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).view(x.size(0), -1)  # flatten

        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        """training step for lightning module"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # calculate acc
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)

        # log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        validation step for lightning module
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5  # , verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class VGG_PSDClassifier(pl.LightningModule):
    """
    VGG_PSDClassifier
    A VGG-like CNN for classifying PSDs
    """

    def __init__(self, num_classes, lr=1e-3, weight_decay=1e-4):
        super(VGG_PSDClassifier, self).__init__()
        self.save_hyperparameters()

        # dimensions for VGG convolutional blocks, in the format (num_convs, out_channels)
        self.convs = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # global average pooling

        # fully connected layers
        self.fc1 = nn.LazyLinear(384)
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.LazyLinear(64)
        self.dropout2 = nn.Dropout(0.35)
        self.fc3 = nn.LazyLinear(num_classes)

        # some other attributes
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc, self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ), torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _vgg_block(self, num_convs, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        """forward pass through the network"""
        x = self.convs(x)
        x = self.gap(x).view(x.size(0), -1)  # flatten
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        """training step for lightning module"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # calculate acc
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)

        # log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        validation step for lightning module
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5  # , verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
