import torch

from rmsnorm import RMSNorm
from llama_block import DecoderLayer

class LLaMA():
    def __init__(self, vocab_size, d_model, n_layers, gamma, n_heads, n_kv_heads, head_dim, max_seq_len, final_gamma):
        self.embedding = torch.ones(vocab_size, d_model)
        decorderList =[]
        for _ in range(n_layers):
            dl = DecoderLayer(d_model,n_heads,n_kv_heads,head_dim,max_seq_len)
            decorderList.append(dl)
        self.decorderList = decorderList
        self.gammaList = gamma
        self.final_rmsnorm = RMSNorm()
        self.final_gamma = final_gamma
        self.LM_head = torch.ones(d_model, vocab_size)
        self.n_layers = n_layers
    def forward(self, x):
        x =  self.embedding[x]
        for i in range(self.n_layers):
            x = self.decorderList[i].forward(x,self.gammaList[i][0], self.gammaList[i][1])
        x = self.final_rmsnorm.forward(x,self.final_gamma)
        x = x @ self.LM_head
        return x
    
    def generate(self, token_ids, max_len, eos_id,temperature=1.0, top_k=0, top_p=0.0):
        
        while len(token_ids) < max_len:
            token_ids_tensor = torch.tensor([token_ids])
            logits = self.forward(token_ids_tensor)
            logits = logits[:,-1,:]
            if temperature == 0:

                next_token = torch.argmax(logits, dim= -1).item()
            else :
                logits = logits / temperature
                if top_k > 0:
                    topk_values, _ =torch.topk(logits, top_k, dim=-1)
                    
                    logits[logits<topk_values[:,-1:]] = float('-inf')
                if top_p > 0:
                    sorted_logits, indices = torch.sort(logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)

                    cutoff_idx = (cumsum > top_p).nonzero(as_tuple=True)[1][0]
                    indices_to_remove = indices[:, cutoff_idx+1:]
                    logits.scatter_(1,indices_to_remove, float('-inf'))
                probs = torch.softmax(logits, dim = -1)
                next_token = torch.multinomial(probs, 1).item()
            if next_token == eos_id:
                break
            token_ids.append(next_token)

        return token_ids
            
