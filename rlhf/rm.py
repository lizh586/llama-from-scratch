import sys
from pathlib import Path
import torch.nn as nn
import torch
LLM = Path(__file__).resolve().parents[1]
sys.path +=[str(LLM/"llama"), str(LLM/"rlhf")]
from llama_model import LLaMA
from rm_batch import pad_mask

class RM(nn.Module):
    def __init__(self, ckpt_path:str):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.config = ckpt["config"]
        self.config["pooling"] = "last"

        # 建 scalar head
        D = self.config["d_model"]
        # WYH: bias 是 chosen 和 rejected 共用的， L对bias求梯度是L对chosen和rejected相加，结果为零
        self.scalar_head = nn.Linear(D,1,bias = False)
        model = LLaMA(ckpt["vocab_size"], self.config["d_model"],self.config["n_layers"],self.config["n_heads"],self.config["n_kv_heads"],self.config["head_dim"],)
        self.model = model
        sd = ckpt["state_dict"]
        if any(k.startswith("scalar_head.") for k in sd):
            self.load_state_dict(sd)
        else:
            self.model.load_state_dict(sd)

        


    def forward(self, input_ids, attn_mask=None):
        h = self.model.get_backbone(input_ids, attn_mask)   # (B, S, D)
        h = h[:, -1, :]                                     # (B, D)
        h = self.scalar_head(h)                             # (B, 1)
        h = h.squeeze(-1)                                   # (B,)
        return h

    def score_pair(self, prompt_ids, response_ids):
        xs = []
        ms = []
        for b in range(len(prompt_ids)):
            ids = prompt_ids[b].tolist() + response_ids[b].tolist()
            x, m = pad_mask(ids)
            xs.append(x)
            ms.append(m)
        x = torch.stack(xs).to(next(self.parameters()).device)
        m = torch.stack(ms).to(next(self.parameters()).device)
        return self.forward(x,m)