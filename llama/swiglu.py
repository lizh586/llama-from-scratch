import torch
import torch.nn.functional as F
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff_gate = int(8 * d_model / 3)
        
        """ self.W_gate = torch.ones(d_model, d_ff_gate)
        self.W_up = torch.ones(d_model, d_ff_gate)
        self.W_down = torch.ones(d_ff_gate, d_model) """

        self.W_gate = nn.Linear(d_model,d_ff_gate, bias=False )
        self.W_up = nn.Linear(d_model, d_ff_gate, bias = False)
        self.W_down = nn.Linear(d_ff_gate, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.W_gate(x))
        up =  self.W_up(x)
        hidden = gate * up
        return self.W_down(hidden)
    
    """ def backward(self, doutput):
        dhidden = doutput @ self.W_down.T
        dW_down = self.hidden.T @ doutput
        dgate = dhidden * self.up
        dup = dhidden * self.gate
        dgate_pre = dgate * (F.sigmoid(self.gate_pre) + self.gate * (1 - F.sigmoid(self.gate_pre)))
        dW_gate = self.x.T @ dgate_pre
        dW_up = self.x.T @ dup
        dx = dgate_pre @ self.W_gate.T + dup @ self.W_up.T
        self.dW_down = dW_down
        self.dW_gate = dW_gate
        self.dW_up = dW_up
        return dx """