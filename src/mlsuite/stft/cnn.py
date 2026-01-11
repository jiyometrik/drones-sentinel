"""
src.mlsuite.stft.cnn
All the CNNs used for STFT classification tasks
"""

import pytorch_lightning as pl
import torch
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

        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)

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


class STFTResNet18(STFTTemplate):
    """A STFT classifier using ResNet architecture"""

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
    """A STFT classifier using VGG11 architecture"""

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


class STFTAlexNet(STFTTemplate):
    """A STFT classifier using AlexNet architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()

        self.alexnet = models.alexnet(pretrained=False)

        self.alexnet.features[0] = nn.Conv2d(
            2,
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


class STFTMobileNetV2(STFTTemplate):
    """A STFT classifier using MobileNetV2 architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()

        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=2,  # mag + phase
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.mobilenet.classifier[1] = nn.Linear(
            self.mobilenet.classifier[1].in_features,
            num_classes,
        )

    def forward(self, x):
        return self.mobilenet(x)


class STFTInceptionV3(STFTTemplate):
    """A STFT classifier using InceptionV3 architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()
        self.inception = models.inception_v3(pretrained=False, aux_logits=False)
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(
            in_channels=2,  # mag + phase
            out_channels=32,
            kernel_size=3,
            stride=2,
            bias=False,
        )
        self.inception.fc = nn.Linear(
            self.inception.fc.in_features,
            num_classes,
        )

    def forward(self, x):
        return self.inception(x)


class STFTDenseNet121(STFTTemplate):
    """A STFT classifier using DenseNet121 architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()
        self.densenet = models.densenet121(pretrained=False)
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=2,  # mag + phase
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.densenet.classifier = nn.Linear(
            self.densenet.classifier.in_features,
            num_classes,
        )

    def forward(self, x):
        return self.densenet(x)
