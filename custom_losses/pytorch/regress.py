import torch
import torch.nn.functional as F

def simple_mse(output, target):
    #loss = torch.mean((output - target)**2, requires_grad = True)
    return torch.nn.MSELoss()(output, target)