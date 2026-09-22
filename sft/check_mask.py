import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SFT = Path(__file__).resolve().parent
sys.path += [str(SFT), str(SFT.parent / "pipeline")]

from data import VOCAB
from sft_data import render, build_input_ids, build_labels

s, n = render('Reverse the input string.', 'abc', 'cba')
ids = build_input_ids(s)
L = len(ids)

y = ids[1:]
labels = build_labels(n, ids)

torch.manual_seed(0)
logits = torch.randn(L - 1, len(VOCAB))
t_y = torch.tensor(y)
t_lab = torch.tensor(labels)

A = F.cross_entropy(logits, t_lab, ignore_index=-100)
C = F.cross_entropy(logits, t_y)

lp = F.log_softmax(logits, dim=-1)
B = torch.stack([-lp[i, y[i]] for i in range(n - 1, L - 1)]).mean()

print(f"n={n}  L={L}  len(labels)={len(labels)}  len(y)={len(y)}")
print(f"response 段在 y 上的下标 = [{n-1}, {L-2}]  共 {L-1-(n-1)} 位")
print()
print(f"A (masked)       = {A.item():.6f}")
print(f"B (手算 resp 段) = {B.item():.6f}")
print(f"C (unmasked)     = {C.item():.6f}")
print()
assert torch.allclose(A, B), f"masked loss ≠ 手算 response 段: {A.item()} vs {B.item()}"
assert not torch.allclose(A, C), "masked 与 unmasked 相同 —— mask 没生效"

print("A == B ? True    (masked = 手算 response 段)")
print("A != C ? True    (mask 确实改变了 loss)")
