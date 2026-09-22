import torch
import torch.nn as nn
from rmsnorm import RMSNorm
from gqa import GroupedQueryAttention
from swiglu import SwiGLU

class DecoderLayer(nn.Module):
    def __init__(self,d_model, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim)
        self.swiglu = SwiGLU(d_model)
        
    def forward(self, x, attn_mask=None):
        x = x + self.gqa(self.attn_norm(x), attn_mask)
        x = x + self.swiglu(self.ffn_norm(x))
        return x

        