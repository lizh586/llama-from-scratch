import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self,x):
        """ diamond = x.shape[-1]
        r = 0
        
        r = r + (x ** 2).mean(dim = -1, keepdim=True)
        
        r = r ** 0.5
        self.r = r
        self.x = x
        self.diamond = diamond

        return x/r * self.gamma """
        r = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x/r * self.gamma
    
    """ def backward(self, grad_output):
        out = self.gamma / self.r * grad_output - self.x / (self.diamond * self.r ** 3) * (self.gamma * self.x * grad_output).sum(dim=-1,keepdim=True)
        return out """
