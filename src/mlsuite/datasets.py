"""
src.mlsuite.datasets
Custom dataset classes for PyTorch
"""

import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.mlsuite import augment


class IFFTDataset(Dataset):
    """
    Custom IFFT data dataset using STFTs
    X: ifft_dict["stft"] (pandas.Series of numpy.ndarray)
    y: ifft_dict["drone_idx"] (pandas.Series of int64)
    """

    def __init__(self, X, y, augment_enabled=False):
        # WARN: what .tolist() does to the dimensions of the tensors is unclear, but prevents errors
        self.X = X.tolist() if isinstance(X, pd.Series) else X
        self.y = y.tolist() if isinstance(y, pd.Series) else y
        self.augment = augment_enabled

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])  # (2, F, T)
        y = torch.LongTensor([self.y[idx]]).squeeze()

        mag = x[0:1]
        phase = x[1:2]

        if self.augment:
            if random.random() < 0.5:
                mag = augment.time_mask(mag)
            if random.random() < 0.5:
                mag = augment.freq_mask(mag)
            if random.random() < 0.5:
                mag = augment.add_noise(mag)
            if random.random() < 0.5:
                mag = augment.amp_scale(mag)

        x = torch.cat([mag, phase], dim=0)

        return x, y


class PSDSequenceDataset(Dataset):
    """
    X: numpy array or list with shape (N, T, F)
    y: labels (N,)
    """

    def __init__(self, X, y):
        self.X = X.tolist() if isinstance(X, pd.Series) else X
        self.y = y.tolist() if isinstance(y, pd.Series) else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.LongTensor([self.y[idx]]).squeeze(),
        )
