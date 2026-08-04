import torch
import torch.nn.functional as F

class SwiGLU():
    def __init__(self, d_model):
        d_ff_gate = int(8 * d_model / 3)
        self.W_gate = torch.ones(d_model, d_ff_gate)
        self.W_up = torch.ones(d_model, d_ff_gate)
        self.W_down = torch.ones(d_ff_gate, d_model)

    def forward(self, x):
        gate_pre = x @ self.W_gate
        self.gate_pre = gate_pre
        gate = F.silu(gate_pre)
        self.gate = gate
        up = x @ self.W_up
        self.up = up
        hidden = gate * up
        self.hidden = hidden
        output = hidden @ self.W_down
        self.x = x
        return output
    
    def backward(self, doutput):
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
        return dx