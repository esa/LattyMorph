import torch

EPS = 1e-7

def sign(scalar):
    return (1.*(scalar >= 0)-0.5)*2

def arccos_clipped(x):
    return torch.arccos(torch.clamp(x, -1 + EPS, 1 - EPS))

def arcsin_clipped(x):
    return torch.arcsin(torch.clamp(x, -1 + EPS, 1 - EPS))