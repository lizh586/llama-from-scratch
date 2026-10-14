"""check_advantage_norm.py — P4-1 advantage / normalization 静默失败点检查（只读；不碰 ckpt / ppo_lm 本体）"""
from pathlib import Path
import sys, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path += [str(Path(__file__).resolve().parent)]
import torch
from ppo_lm import advantage, normalization

FAIL = []


def report(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<26} {detail}")
    if not ok:
        FAIL.append(tag)


def hand_suffix_sum(rows):
    """独立实现：python float，倒着累加，不走 torch"""
    out = []
    for r in rows:
        acc, row = 0.0, [0.0] * len(r)
        for s in range(len(r) - 1, -1, -1):
            acc += r[s]
            row[s] = acc
        out.append(row)
    return out


def hand_mean_std(vals):
    """无偏（除 n-1），与 torch.std 默认一致"""
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


# ---------- A advantage 手算对照 ----------
reward = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [0.5, -1.0, 2.0, 0.0]])
value = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                      [-0.5, 0.0, 1.0, 0.2]])
adv = advantage(reward, value)
ref = torch.tensor(hand_suffix_sum(reward.tolist())) - value
dA = (adv - ref).abs().max().item()
report("A 手算对照(后缀和)", dA <= 1e-6, f"max|Δ| = {dA:.2e}   adv={adv.tolist()}")

# ---------- A' 三种坏法必须对不上 ----------
bad_adv = {
    "只 flip 一次": torch.cumsum(torch.flip(reward, dims=[-1]), dim=-1) - value,
    "正向 cumsum": torch.cumsum(reward, dim=-1) - value,
    "少一个位": torch.flip(torch.cumsum(torch.flip(reward, dims=[-1]), dim=-1), dims=[-1]).roll(1, -1) - value,
}
for name, b in bad_adv.items():
    db = (b - ref).abs().max().item()
    report(f"A' 反证·{name}", db > 1e-3, f"max|Δ vs 手算| = {db:.3e}")

# ---------- B shape / dtype ----------
report("B shape 同位", tuple(adv.shape) == tuple(reward.shape), f"{tuple(adv.shape)} vs {tuple(reward.shape)}")
report("B dtype 浮点", adv.is_floating_point(), f"{adv.dtype}")

# ---------- C normalization 手算对照 ----------
A = torch.tensor([[9.0, 1.0, 3.0, 5.0, 2.0],
                  [4.0, 8.0, 6.0, 0.0, 7.0]])
mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0],
                     [0.0, 1.0, 1.0, 1.0, 0.0]])
eps = 1e-8
out = normalization(A, eps, mask)
vals = [A[b][s].item() for b in range(A.shape[0]) for s in range(A.shape[1]) if mask[b][s].item() == 1.0]
mu, sd = hand_mean_std(vals)
ref_n = (A - mu) / (sd + eps)
dC = (out - ref_n).abs().max().item()
report("C 手算对照(μ/σ)", dC <= 1e-6, f"max|Δ| = {dC:.2e}   μ={mu:.6f} σ={sd:.6f}")

# ---------- C' 反证：少括号（优先级） ----------
bad_prec = A - mu / (sd + eps)
dCb = (bad_prec - ref_n).abs().max().item()
report("C' 反证·少括号", dCb > 1e-3, f"max|Δ vs 手算| = {dCb:.3e}")

# ---------- D 统计域只在 mask 内 ----------
A2 = A.clone()
A2[mask == 0] = 1e6                    # mask 外的值放大，μ/σ 不应受影响
out2 = normalization(A2, eps, mask)
dD = (out2[mask.bool()] - ref_n[mask.bool()]).abs().max().item()
report("D 统计域=mask 内", dD <= 1e-6, f"mask 内 max|Δ| = {dD:.2e}")
bad_all = (A2 - A2.mean()) / (A2.std() + eps)      # 反证：整块统计
dDb = (bad_all[mask.bool()] - ref_n[mask.bool()]).abs().max().item()
report("D' 反证·整块统计", dDb > 1e-3, f"mask 内 max|Δ| = {dDb:.3e}")

# ---------- E μ/σ detached ----------
c = sd + eps
A3 = A.clone().requires_grad_(True)
normalization(A3, eps, mask).sum().backward()
dE = (A3.grad - 1.0 / c).abs().max().item()
report("E μ/σ detached", dE <= 1e-6, f"max|grad − 1/(σ+ε)| = {dE:.2e}")


def norm_nodetach(A_t, epsilon, resp_mask):
    sel = A_t[resp_mask.bool()]
    return (A_t - sel.mean()) / (sel.std() + epsilon)


A4 = A.clone().requires_grad_(True)
norm_nodetach(A4, eps, mask).sum().backward()
dEb = (A4.grad - 1.0 / c).abs().max().item()
report("E' 反证·不 detach", dEb > 1e-6, f"max|grad 差| = {dEb:.3e}")

# ---------- F σ=0 边界 ----------
A5 = torch.zeros((2, 5))
A5[:, 1:3] = 3.0
mask5 = torch.zeros((2, 5))
mask5[:, 1:3] = 1.0
out5 = normalization(A5, eps, mask5)
report("F σ=0 不出 NaN", bool(torch.isfinite(out5).all()), f"finite = {bool(torch.isfinite(out5).all())}")
report("F σ=0 mask 内=0", float(out5[mask5.bool()].abs().max()) == 0.0, f"mask 内 max|out| = {out5[mask5.bool()].abs().max().item():.1e}")

# ---------- G shape 同位 ----------
report("G shape 同位", tuple(out.shape) == tuple(A.shape), f"{tuple(out.shape)} vs {tuple(A.shape)}")

print()
print(f"== {len(FAIL)} FAIL: {FAIL} ==" if FAIL else "== ALL PASS ==")
