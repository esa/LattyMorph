import torch

def rotate_y(angle):
    res = torch.Tensor([[0,0,0],[0,0,0],[0,0,0]])
    res[0][0] = torch.cos(angle) 
    res[0][2] = torch.sin(angle)
    res[1][1] = 1
    res[2][0] = -torch.sin(angle)
    res[2][2] = torch.cos(angle)
    
    return res
    
def rotate_z(angle):
    res = torch.Tensor([[0,0,0],[0,0,0],[0,0,0]])
    res[0][0] = torch.cos(angle) 
    res[0][1] = -torch.sin(angle)
    res[2][2] = 1
    res[1][0] = torch.sin(angle)
    res[1][1] = torch.cos(angle)
    
    return res
    
def rotate_x(angle):
    res = torch.Tensor([[0,0,0],[0,0,0],[0,0,0]])
    res[1][1] = torch.cos(angle) 
    res[1][2] = -torch.sin(angle)
    res[0][0] = 1
    res[2][1] = torch.sin(angle)
    res[2][2] = torch.cos(angle)
    
    return res