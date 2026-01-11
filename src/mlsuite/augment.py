"""
src.mlsuite.augment
All the data augmentation utilities for ML tasks
"""

import random

import torch


def time_mask(x, max_width=12):
    """Mask the input tensor x along the time axis."""
    x = x.clone()
    _, _, T = x.shape
    w = random.randint(0, min(max_width, T // 4))
    if w == 0:
        return x
    t0 = random.randint(0, T - w)
    x[:, :, t0 : t0 + w] = 0
    return x


def freq_mask(x, max_width=10):
    """Mask the input tensor x along the frequency axis."""
    x = x.clone()
    _, F, _ = x.shape
    w = random.randint(0, min(max_width, F // 4))
    if w == 0:
        return x
    f0 = random.randint(0, F - w)
    x[:, f0 : f0 + w, :] = 0
    return x


def add_noise(x, std=0.01):
    """Add Gaussian noise to the input tensor x."""
    noise = torch.randn_like(x) * (std * x.std())
    return x + noise


def amp_scale(x, low=0.8, high=1.2):
    """Scale the amplitude of the input tensor x."""
    return x * random.uniform(low, high)


def freq_shift(x, max_shift=2):
    """Shift the input tensor x along the frequency axis."""
    shift = random.randint(-max_shift, max_shift)
    if shift != 0:
        x = torch.roll(x, shifts=shift, dims=1)
    return x
