"""
src.mlsuite.stft.cnn
All the CNNs used for STFT classification tasks
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from torch import nn

import src.constants as cts


class STFTTemplate(pl.LightningModule):
    """
    A template class for STFT classifiers using CNN architectures;
    all subsequent STFT classifier classes should inherit from this one
    """

    def __init__(self, num_classes):
        super(STFTTemplate, self).__init__()
        self.save_hyperparameters()

        # Universal CNN attributes
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch):
        """training step for lightning module"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # calculate acc
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch):
        """validation step for lightning module"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

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


class STFTResNet18(STFTTemplate):
    """An STFT classifier using ResNet architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()
        # load a pre-defined ResNet model
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class STFTVgg11(STFTTemplate):
    """An STFT classifier using VGG11 architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()
        # load a pre-defined VGG11 model
        self.vgg11 = models.vgg11(pretrained=False)
        self.vgg11.features[0] = nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False)
        self.vgg11.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        """modified forward pass using VGG11 instead of existing model architecture"""
        return self.vgg11(x)
