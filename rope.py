import torch

class RoPE():
    def __init__(self, head_dim ):
        freq = torch.zeros(head_dim // 2)
        for i in range(head_dim // 2):
            freq[i] = 1.0 / (10000 ** (2 * i / head_dim))
        self.freq = freq.reshape(1, -1)

    def forward(self, x):
        B, L, n_heads, head_dim = x.shape
        x = x.reshape(B, L, n_heads, head_dim//2, 2)
        x_even = x[:,:,:,:,0]
        x_odd  = x[:,:,:,:,1]
        position = torch.arange(L, dtype=torch.float32).reshape(-1, 1)
        angle = position * self.freq  # (L, head_dim//2)
        cos = angle.cos().reshape(1, L, 1, head_dim//2)
        sin = angle.sin().reshape(1, L, 1, head_dim//2)
        _x_even = x_even * cos - x_odd * sin
        _x_odd  = x_even * sin + x_odd * cos
        self.cos = cos
        self.sin = sin
        return torch.stack([_x_even, _x_odd], dim=-1).reshape(B, L, n_heads, head_dim)
    
    def backward(self,grad_output):
        B, L, n_heads, head_dim = grad_output.shape
        grad_output = grad_output.reshape(B, L, n_heads, head_dim//2, 2)
        grad_output_even = grad_output[:,:,:,:,0]
        grad_output_odd  = grad_output[:,:,:,:,1]
        _grad_output_even = grad_output_even * self.cos + grad_output_odd * self.sin
        _grad_output_odd  = -grad_output_even * self.sin + grad_output_odd * self.cos
        return torch.stack([_grad_output_even, _grad_output_odd], dim=-1).reshape(B, L, n_heads, head_dim)
        





