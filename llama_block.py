import torch
from rmsnorm import RMSNorm
from gqa import GroupedQueryAttention
from swiglu import SwiGLU

class DecoderLayer():
    def __init__(self,d_model, n_heads, n_kv_heads, head_dim, max_seq_len):
        self.attn_norm = RMSNorm()
        self.ffn_norm = RMSNorm()
        self.gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, max_seq_len)
        self.swiglu = SwiGLU(d_model)
        
    def forward(self, x, gamma_attn, gamma_ffn):
        out = self.attn_norm.forward(x, gamma_attn)
        x = self.gqa.forward(out) + x
        out = self.ffn_norm.forward(x, gamma_ffn)
        x = self.swiglu.forward(out) + x
        return x

        