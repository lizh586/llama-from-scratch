import torch
import torch.nn as nn

from rmsnorm import RMSNorm
from llama_block import DecoderLayer


class LLaMA(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        """decorderList =[]
        for _ in range(n_layers):
            dl = DecoderLayer(d_model,n_heads,n_kv_heads,head_dim,max_seq_len)
            decorderList.append(dl)
        self.decorderList = decorderList """

        self.decorderList = nn.ModuleList([
            DecoderLayer(d_model, n_heads, n_kv_heads, head_dim)
            for _ in range(n_layers)
        ])
        #self.gammaList = gamma
        self.final_rmsnorm = RMSNorm(d_model)
        # self.LM_head = torch.ones(d_model, vocab_size)
        self.LM_head = nn.Linear(d_model, vocab_size, bias=False)
        self.LM_head.weight = self.embedding.weight
        self.n_layers = n_layers

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attn_mask=None):
        x =  self.embedding(x)
        """ for i in range(self.n_layers):
            x = self.decorderList[i].forward(x,self.gammaList[i][0], self.gammaList[i][1]) """
        for layer in self.decorderList:
            x = layer(x, attn_mask)
        x = self.final_rmsnorm(x)
        logits = self.LM_head(x)
        return logits
    def get_backbone(self, x, attn_mask=None):
        x =  self.embedding(x)
        """ for i in range(self.n_layers):
            x = self.decorderList[i].forward(x,self.gammaList[i][0], self.gammaList[i][1]) """
        for layer in self.decorderList:
            x = layer(x, attn_mask)
        h = self.final_rmsnorm(x)

        return h


    @torch.no_grad()
    def generate(self, token_ids, max_len, eos_id,temperature=1.0, top_k=0, top_p=0.0, attn_mask=None):
        if not torch.is_tensor(token_ids) or token_ids.dim() == 1:
            out = list(token_ids)
            while len(out) < max_len:
                token_ids_tensor = torch.tensor([out],device=next(self.parameters()).device)
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
                out.append(next_token)

            return out
        else:
            out = token_ids
            B, _ = out.shape
            finished = torch.full((B,), False, dtype=torch.bool, device=out.device)

            while out.shape[1] < max_len and not finished.all():
                logits = self.forward(out,attn_mask=attn_mask)

                logits = logits[:,-1,:]
                if temperature == 0 :
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        topk_values, _ = torch.topk(logits,top_k, dim=-1)
                        logits[logits<topk_values[:,-1,:]] = float('-inf')
                    if top_p > 0:
                        sorted_logits, indices = torch.sort(logits, descending=True)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumsum = torch.cumsum(sorted_probs, dim=-1)
                        # 每行第一个 ( cumsum > p ) 的 "排序位次" -> (B, 1)
                        cutoff_idx = (cumsum > top_p).float().argmax(dim=-1, keepdim=True)
                        rank = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
                        keep_rank = rank <= cutoff_idx
                        keep_mask = torch.zeros_like(logits).scatter_(1,indices, keep_rank.float())
                        logits = logits.masked_fill(keep_mask == 0, float('-inf'))
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                finished = finished | (next_token == eos_id).squeeze(1)
                out = torch.cat([out, next_token], dim=1)
                attn_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=attn_mask.dtype, device=out.device)], dim=1,)
            return out
            
