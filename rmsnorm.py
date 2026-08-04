import torch

class RMSNorm():
    def forward(self,x, gamma):
        diamond = x.shape[-1]
        r = 0
        
        r = r + (x ** 2).mean(dim = -1, keepdim=True)
        
        r = r ** 0.5
        self.r = r
        self.gamma = gamma
        self.x = x
        self.diamond = diamond
        return x/r * gamma
    
    def backward(self, grad_output):
        out = self.gamma / self.r * grad_output - self.x / (self.diamond * self.r ** 3) * (self.gamma * self.x * grad_output).sum(dim=-1,keepdim=True)
        return out
