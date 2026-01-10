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
import torchmetrics
from torchmetrics.classification import MulticlassF1Score
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
        
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

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
        f1 = self.train_f1(preds, labels)

        # log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        """validation step for lightning module"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1", f1, prog_bar=True, on_step=False, on_epoch=True)
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
    
class STFTAlexNet(STFTTemplate):
    """An STFT classifier using AlexNet architecture"""

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
    """An STFT classifier using MobileNetV2 architecture"""

    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.save_hyperparameters()

        self.mobilenet = models.mobilenet_v2(pretrained=False)

        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=2,    # mag + phase
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
