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


class STFTSimple(pl.LightningModule):
    """a simple 2D CNN for classifying STFTs"""

    def __init__(self, num_classes, lr=1e-3, weight_decay=1e-4):
        super(STFTSimple, self).__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 5, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 512, 5, 2)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.35)
        self.fc3 = nn.Linear(128, num_classes)

        # some other attributes
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc, self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ), torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """forward pass through the network"""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # flatten
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
        """validation step for lightning module"""
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


class STFTResNet(STFTSimple):
    """an STFT classifier using ResNet architecture"""

    def __init__(self, num_classes, lr=1e-3, weight_decay=1e-4, transfer_learning=True):
        super().__init__(num_classes, lr, weight_decay)
        self.save_hyperparameters()
        # load a pre-defined ResNet model
        self.resnet = models.resnet50(pretrained=transfer_learning)
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc, self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ), torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """modified forward pass using ResNet instead of existing model architecture"""
        return self.resnet(x)


class STFTVgg11(STFTSimple):
    """an STFT classifier using VGG11 architecture"""

    def __init__(self, num_classes, lr=1e-3, weight_decay=1e-4):
        super().__init__(num_classes, lr, weight_decay)
        self.save_hyperparameters()
        # load a pre-defined VGG11 model
        self.vgg11 = models.vgg11(weights="default")
        self.vgg11.features[0] = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.vgg11.classifier[-1] = nn.Linear(4096, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc, self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ), torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """modified forward pass using VGG11 instead of existing model architecture"""
        return self.vgg11(x)
