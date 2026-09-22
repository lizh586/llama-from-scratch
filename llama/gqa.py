import torch
import torch.nn as nn
from rope import RoPE

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim):
        """ self.W_q = torch.ones(d_model, n_heads * head_dim)
        self.W_k = torch.ones(d_model, n_kv_heads * head_dim)
        self.W_v = torch.ones(d_model, n_kv_heads * head_dim)
        self.W_o = torch.ones(n_heads * head_dim, d_model)
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim """

        super().__init__()
        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * head_dim, bias= False)
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope = RoPE(head_dim)
        

    def repeat_kv(self, x):
        return torch.repeat_interleave(x, repeats=(self.n_heads // self.n_kv_heads), dim=2)
    

    def forward(self, x, attn_mask=None):
        batch, seq, _ = x.shape

        """ Q = (x @ self.W_q).reshape(batch, seq, self.n_heads, self.head_dim)
        K = (x @ self.W_k).reshape(batch, seq, self.n_kv_heads, self.head_dim)
        V = (x @ self.W_v).reshape(batch, seq, self.n_kv_heads, self.head_dim) """
        Q = self.W_q(x).reshape(batch, seq, self.n_heads, self.head_dim)
        K = self.W_k(x).reshape(batch, seq, self.n_kv_heads, self.head_dim)
        V = self.W_v(x).reshape(batch, seq, self.n_kv_heads, self.head_dim)

        Q = self.rope(Q)
        K = self.rope(K)
        Q = Q.transpose(1,2)

        K = self.repeat_kv(K)
        V = self.repeat_kv(V)
        K = K.transpose(1,2)
        V = V.transpose(1,2)        
        scores = Q  @ K.transpose(-2, -1) / self.head_dim ** 0.5
        casual_mask = torch.triu(torch.full((seq, seq), torch.finfo(scores.dtype).min, device=scores.device), diagonal=1)
        scores = scores + casual_mask
        # attn_mask 一个batch里的序列长度不同，需要补齐到等长，屏蔽补出来的假token
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.bool()[:,None, None,:],torch.finfo(scores.dtype).min)
        attn_weights = torch.softmax(scores, dim = -1)
        output = attn_weights @ V
        output = output.transpose(1,2).reshape(batch,seq,self.n_heads * self.head_dim)
        return self.W_o(output)


