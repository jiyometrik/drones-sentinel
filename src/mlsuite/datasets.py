"""
src.mlsuite.datasets
custom dataset classes for PyTorch
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class PSDDataset(Dataset):
    """
    custom IFFT data dataset using Welch's PSDs
    X: ifft_dict["psd"] (pandas.Series of numpy.ndarray)
    y: ifft_dict["drone_idx"] (pandas.Series of int64)
    """

    def __init__(self, X, y):
        # WARN: what .tolist() does to the dimensions of the tensors is unclear, but prevents errors
        self.X = X.tolist() if isinstance(X, pd.Series) else X
        self.y = y.tolist() if isinstance(y, pd.Series) else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # only return torch.Tensors when called
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]]).squeeze()
