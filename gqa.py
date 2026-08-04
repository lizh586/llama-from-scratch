import torch
from rope import RoPE

class GroupedQueryAttention():
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim, max_seq_len):
        self.W_q = torch.ones(d_model, n_heads * head_dim)
        self.W_k = torch.ones(d_model, n_kv_heads * head_dim)
        self.W_v = torch.ones(d_model, n_kv_heads * head_dim)
        self.W_o = torch.ones(n_heads * head_dim, d_model)
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    def repeat_kv(self, x):
        return torch.repeat_interleave(x, repeats=(self.n_heads // self.n_kv_heads), dim=2)
    

    def forward(self, x):
        batch, seq, d_model = x.shape
        Q = (x @ self.W_q).reshape(batch, seq, self.n_heads, self.head_dim)
        K = (x @ self.W_k).reshape(batch, seq, self.n_kv_heads, self.head_dim)
        V = (x @ self.W_v).reshape(batch, seq, self.n_kv_heads, self.head_dim)
        rope = RoPE(self.head_dim)
        Q = rope.forward(Q)
        K = rope.forward(K)
        Q = Q.transpose(1,2)

        K = self.repeat_kv(K)
        V = self.repeat_kv(V)
        K = K.transpose(1,2)
        V = V.transpose(1,2)        
        scores = Q  @ K.transpose(-2, -1) / self.head_dim ** 0.5
        casual_mask = torch.triu(torch.full((seq, seq), float('-inf')), diagonal=1)
        scores = scores + casual_mask
        attn_weights = torch.softmax(scores, dim = -1)
        output = attn_weights @ V
        output = output.transpose(1,2).reshape(batch,seq,self.n_heads * self.head_dim)
        output = output @ self.W_o
        return output 


