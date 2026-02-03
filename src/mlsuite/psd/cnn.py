"""
src.mlsuite.psd.cnn
All the CNNs used for Welch PSD (image) classification tasks
"""

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.models as models
from torch import nn

import src.constants as cts

INPUT_CHANNELS = 16


class PSDTemplate(pl.LightningModule):
    """
    A template class for PSD classifiers using CNN architectures;
    all subsequent PSD classifier classes should inherit from this one
    """

    def __init__(self, num_classes):
        super(PSDTemplate, self).__init__()
        self.save_hyperparameters()

        # Universal CNN attributes
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_classes, average="macro"
        )

    def training_step(self, batch, batch_idx):
        """training step for LightningModule"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # calculate metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        f1 = self.train_f1(preds, labels)

        # log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        """validation step for LightningModule"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # calculate metrics
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)

        # log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)
        f1 = self.test_f1(preds, labels)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=cts.LR,
            weight_decay=cts.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
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


class PSDAlexNet(PSDTemplate):
    """A PSD classifier using AlexNet architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()

        self.alexnet = models.alexnet(pretrained=False)

        self.alexnet.features[0] = nn.LazyConv2d(
            64,
            kernel_size=11,
            stride=4,
            padding=2,
        )

        self.alexnet.classifier[-1] = nn.Linear(
            self.alexnet.classifier[-1].in_features,
            num_classes,
        )

    def forward(self, x):
        return self.alexnet(x)
