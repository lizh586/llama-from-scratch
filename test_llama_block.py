import torch
from llama_block import DecoderLayer

d_model = 4096
n_heads = 32
n_kv_heads = 8
head_dim = 128
max_seq_len = 2048

layer = DecoderLayer(d_model, n_heads, n_kv_heads, head_dim, max_seq_len)

batch, seq = 2, 8
x = torch.randn(batch, seq, d_model)
gamma_attn = torch.ones(d_model)
gamma_ffn = torch.ones(d_model)

out = layer.forward(x, gamma_attn, gamma_ffn)

assert out.shape == (batch, seq, d_model), f"Expected {(batch, seq, d_model)}, got {out.shape}"
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print("DecoderLayer forward: shape OK")
