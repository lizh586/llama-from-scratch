"""P3 出口④ 的检查：compute_logprobs 的硬判据。

判据（每条都能失败）：
  A. 手算对照：逐位置、显式下标实现 log pi = z[a] - logsumexp(z)
     （不碰 log_softmax / cross_entropy），与函数输出一致
     —— 符号 / 错位方向 / 拼装，任一处错这里必炸
  A'. 反证：把符号翻掉、把错位方向翻掉，各自必须与函数输出不一致
     —— 对不上才说明 A 有分辨力
  B. 契约：shape == ids.shape；[:,0] 恒 0；dtype 浮点
  C. 梯度：out[:,1:].sum().backward() 后参数有非零 grad（P4 actor 路要用）
  D. mask 透传：左 pad 批的 logprob == 该行单独跑（不 pad）的 logprob（平移对位）
  D'. 反证：全 1 mask 下 D 的读数必须变
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "llama"))
sys.path.insert(0, str(HERE))

from llama_model import LLaMA  # noqa: E402
from ppo_lm import compute_logprobs, PAD_ID  # noqa: E402

torch.manual_seed(0)
model = LLaMA(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16)
model.eval()

ids = torch.tensor([[10, 11, 12, 13],
                    [20, 21, 22, 23]])
B, S = ids.shape

out = compute_logprobs(model, ids)


def manual_lp(z, ids):
    """逐位置手算：token t 的 logprob 取 z 的 t-1 位，显式下标 + 手写 logsumexp。"""
    B, S, _ = z.shape
    lp = torch.zeros(B, S)
    rows = torch.arange(B)
    for t in range(1, S):
        zt = z[:, t - 1, :]
        m = zt.max(dim=-1, keepdim=True).values
        lse = m + (zt - m).exp().sum(dim=-1, keepdim=True).log()
        lp[:, t] = zt[rows, ids[:, t]] - lse.squeeze(1)
    return lp


with torch.no_grad():
    z = model(ids)

# ---- A: 手算对照 ----
manual = manual_lp(z, ids)
dA = (out - manual).abs().max().item()
print(f"A  manual max|d| = {dA:.2e}")
assert dA < 1e-5, "判据 A FAIL：与逐位置手算不一致（符号/错位/拼装）"

# ---- A': 反证 —— 翻符号 / 翻错位方向，必须对不上 ----
d_sign = (out - (-manual)).abs().max().item()

wrong = torch.zeros(B, S)
rows = torch.arange(B)
for t in range(0, S - 1):
    zt = z[:, t + 1, :]
    m = zt.max(dim=-1, keepdim=True).values
    lse = m + (zt - m).exp().sum(dim=-1, keepdim=True).log()
    wrong[:, t] = zt[rows, ids[:, t]] - lse.squeeze(1)
band = slice(1, S - 1)
d_shift = (out[:, band] - wrong[:, band]).abs().max().item()

print(f"A' control  sign-flip max|d| = {d_sign:.2e} | shift-flip max|d| = {d_shift:.2e}")
assert d_sign > 1e-3, "反证 FAIL：翻符号后居然还对得上 —— A 没分辨力"
assert d_shift > 1e-3, "反证 FAIL：翻错位后居然还对得上 —— A 没分辨力"

# ---- B: 契约 ----
assert tuple(out.shape) == (B, S), f"shape 应为 {(B, S)}，实际 {tuple(out.shape)}"
assert out.dtype.is_floating_point, "dtype 必须是浮点"
assert (out[:, 0] == 0).all(), "[:,0] 必须为 0（无前驱）"
print("B  contract PASS: shape / dtype / col0==0")

# ---- C: 梯度流通 ----
model.zero_grad()
out_g = compute_logprobs(model, ids)
out_g[:, 1:].sum().backward()
n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
assert n_grad > 0, "判据 C FAIL：backward 后没有任何参数拿到梯度"
print(f"C  grad PASS: {n_grad} params with nonzero grad")
model.zero_grad()

# ---- D: mask 透传 —— 左 pad 批 vs 单跑（平移对位）----
prompts = [[10, 11, 12], [20, 21, 22, 23]]
L0 = max(len(p) for p in prompts)
xb = torch.full((len(prompts), L0), PAD_ID, dtype=torch.long)
mb = torch.zeros((len(prompts), L0), dtype=torch.long)
for i, p in enumerate(prompts):
    xb[i, L0 - len(p):] = torch.tensor(p)
    mb[i, L0 - len(p):] = 1

with torch.no_grad():
    lp_batch = compute_logprobs(model, xb, attn_mask=mb)
    lp_ones = compute_logprobs(model, xb, attn_mask=torch.ones_like(mb))
    lp_single = [compute_logprobs(model, torch.tensor([p])) for p in prompts]

dmax = 0.0
for i, p in enumerate(prompts):
    off = L0 - len(p)
    for k in range(1, len(p)):
        d = abs(lp_batch[i, k + off].item() - lp_single[i][0, k].item())
        dmax = max(dmax, d)
print(f"D  left-pad vs single max|d| = {dmax:.2e}")
assert dmax < 1e-5, "判据 D FAIL：左 pad 批与单跑的 logprob 不一致（mask 没透传？）"

d_ones = (lp_batch - lp_ones).abs().max().item()
print(f"D' control  all-ones max|d| = {d_ones:.2e}")
assert d_ones > 1e-3, "反证 FAIL：全 1 mask 没变 —— D 是空检查"

print("\nALL PASS")
