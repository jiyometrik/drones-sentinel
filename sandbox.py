"""
testrun.py
The main script for running the helper files.
To be used to test out difference classification model architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sourdough as sd
import spectrogram as spectro
from mlsuite import cnn1d, lenet, random_forest, starter

# want rf? use -1
FIXED_LENGTH = 4096
LR = 1e-3

# Preprocess all IFFT files
# sd.unload_zip_files(zipdir=sd.ZIPDIR, extractdir=sd.DATADIR)
# X, y = sd.get_features(datadir=sd.DATADIR, fixed_length=FIXED_LENGTH)
spectrograms, labels = spectro.get_spectrograms(
    datadir=sd.DATADIR,
    spectro_dir=spectro.SPECTRODIR,
    fs=1000,  # Adjust based on your sampling frequency
    image_size=(224, 224),  # Common size for image classification
    save_images=True,
    method="stft",  # or 'cwt' for continuous wavelet transform
)

# Normalize spectrograms to [0, 1] range
X_normalized = spectrograms.astype(np.float32) / 255.0

helper = starter.Starter(X_normalized, labels)
helper.create_datasets()

train_ds, test_ds = helper.train_dataset, helper.test_dataset
train_loader, test_loader = helper.train_loader, helper.test_loader
n_classes, n_epochs = helper.num_classes, helper.num_epochs

device = helper.device

print(f"{train_ds.__len__ = }")

# --- PLAYGROUND: change only the variables here ---
model = lenet.LeNet(num_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Training sequence ---
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

# --- Evaluation sequence ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")
