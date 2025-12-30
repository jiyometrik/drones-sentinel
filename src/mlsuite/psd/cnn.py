"""
src.mlsuite.psd.cnn
All the CNNs used for PSD classification tasks
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class PSDClassifier1D(pl.LightningModule):
    """A simple 1D CNN for classifying PSDs, using only Conv1d and Linear layers"""

    def __init__(self, num_classes, lr=1e-3, weight_decay=1e-4):
        super(PSDClassifier1D, self).__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(2, 64, kernel_size=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, num_classes)

        # some other attributes
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc, self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ), torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """forward pass through the network"""
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))
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
