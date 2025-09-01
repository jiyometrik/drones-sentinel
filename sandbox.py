"""
testrun.py
The main script for running the helper files.
To be used to test out difference classification model architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import sourdough as sd
from mlsuite import cnn1d, random_forest, starter

# want rf? use -1
FIXED_LENGTH = 4096
LR = 1e-3

# Preprocess all IFFT files
sd.unload_zip_files(zipdir=sd.ZIPDIR, extractdir=sd.DATADIR)
X, y = sd.get_features(datadir=sd.DATADIR, fixed_length=FIXED_LENGTH)

# Get ready for some deep learning
helper = starter.Starter(X, y, num_epochs=50)
helper.create_datasets()

train_ds, test_ds = helper.train_dataset, helper.test_dataset
train_loader, test_loader = helper.train_loader, helper.test_loader
n_classes, n_epochs = helper.num_classes, helper.num_epochs

device = helper.device

# --- PLAYGROUND: change only the variables here ---
model = cnn1d.CNN1D(input_length=FIXED_LENGTH, num_classes=n_classes)
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
