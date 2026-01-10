import torch
import random

def time_mask(x, max_width=12):
    x = x.clone()
    _, _, T = x.shape
    w = random.randint(0, min(max_width, T // 4))
    if w == 0:
        return x
    t0 = random.randint(0, T - w)
    x[:, :, t0:t0 + w] = 0
    return x

def freq_mask(x, max_width=10):
    x = x.clone()
    _, F, _ = x.shape
    w = random.randint(0, min(max_width, F // 4))
    if w == 0:
        return x
    f0 = random.randint(0, F - w)
    x[:, f0:f0 + w, :] = 0
    return x

def add_noise(x, std=0.01):
    noise = torch.randn_like(x) * (std * x.std())
    return x + noise

def amp_scale(x, low=0.8, high=1.2):
    return x * random.uniform(low, high)

def freq_shift(x, max_shift=2):
    shift = random.randint(-max_shift, max_shift)
    if shift != 0:
        x = torch.roll(x, shifts=shift, dims=1)
    return x
