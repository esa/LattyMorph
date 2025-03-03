import torch
from LattyMorph.math.utils import sign, arccos_clipped, arcsin_clipped

def polar_y(angle_z, angle_y):
    res = torch.Tensor([0,0,0])
    res[0] = torch.cos(angle_y)*torch.cos(angle_z)
    res[1] = torch.sin(angle_z)
    res[2] = -torch.sin(angle_y)*torch.cos(angle_z)
    
    return res

def invert_polar_y(vector):
    full_norm = vector.norm()
    partial_norm = vector[::2].norm()
    theta = arcsin_clipped(vector[1]/full_norm)
    phi = -sign(vector[2])*arccos_clipped(vector[0]/partial_norm)

    return full_norm, theta, phi

def polar_x(angle_z, angle_x):
    res = torch.Tensor([0,0,0])
    res[0] = torch.cos(angle_x)*torch.sin(angle_z)
    res[1] = torch.sin(angle_x)*torch.sin(angle_z)
    res[2] = torch.cos(angle_z)
    
    return res

def invert_polar_x(vector):
    full_norm = vector.norm()
    partial_norm = vector[:-1].norm()
    theta = arccos_clipped(vector[2]/full_norm)
    phi = sign(vector[1])*arccos_clipped(vector[0]/partial_norm)
    
    return full_norm, theta, phi

def unitVector(angle):
    res = torch.Tensor([0,0])
    res[0] = torch.cos(angle)
    res[1] = torch.sin(angle)
    return res