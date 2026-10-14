"""check_kl_k3.py — compute_kl_k3 的静默失败点检查（只读；不碰 ckpt / RM / ppo_lm 本体）"""
from pathlib import Path
import sys, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path += [str(Path(__file__).resolve().parent)]
import torch
from ppo_lm import compute_kl_k3

FAIL = []


def report(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<22} {detail}")
    if not ok:
        FAIL.append(tag)


def manual_k3(lt, lr, clamp=50.0):
    """独立实现：python float + math.exp，不走 torch"""
    d = min(lr - lt, clamp)
    return math.exp(d) - d - 1.0


# ---------- A 手算对照 ----------
lt = torch.tensor([[-3.0, -0.5, -1.2]])
lr = torch.tensor([[-1.5, -2.0, -1.2]])
m = torch.ones((1, 3))
out = compute_kl_k3(lt, lr, m)
ref = torch.tensor([[manual_k3(lt[0, i].item(), lr[0, i].item()) for i in range(3)]])
dA = (out - ref).abs().max().item()
report("A 手算对照", dA <= 1e-6, f"max|Δ| = {dA:.2e}   (out={out.tolist()}  hand={ref.tolist()})")

# ---------- A' 三种坏法必须对不上 ----------
d = lr - lt
bad = {
    "漏 -Δ项": torch.exp(d) - 1,
    "翻 Δ 符号": torch.exp(-d) - (-d) - 1,
    "漏 -1": torch.exp(d) - d,
}
for name, b in bad.items():
    db = (b - ref).abs().max().item()
    report(f"A' 反证·{name}", db > 1e-3, f"max|Δ vs 手算| = {db:.3e}")

# ---------- B 非负 ----------
d = torch.linspace(-60, 60, 2001).reshape(1, -1)
k = compute_kl_k3(torch.zeros_like(d), d.clone(), torch.ones_like(d))
report("B 非负 Δ∈[-60,60]", bool((k >= 0).all()), f"min k3 = {k.min().item():.3e}")

b1 = 1 + d - torch.exp(d)          # k3 -> -k3（整式翻号）
b2 = -d                            # 写成 k1 = -Δ
report("B' 反证·翻号", bool((b1 < 0).any()), f"min = {b1.min().item():.3e}  (应 <0)")
report("B'' 反证·写成k1", bool((b2 < 0).any()), f"min = {b2.min().item():.3e}  (应 <0)")

# ---------- C clamp 生效 ----------
c_out = compute_kl_k3(torch.tensor([[-80.0]]), torch.tensor([[0.0]]), torch.ones((1, 1))).item()
c_ref = math.exp(50) - 50 - 1
c_bad = math.exp(80) - 80 - 1
report("C clamp 生效 (Δ=80)", abs(c_out - c_ref) <= 1e-5 * c_ref, f"out = {c_out:.6e}   e^50-51 = {c_ref:.6e}")
report("C' 反证·去 clamp", c_bad != c_ref, f"无 clamp = {c_bad:.6e}   (差 {c_bad - c_ref:.3e})")

# ---------- D mask 生效 ----------
lt4 = torch.tensor([[-3.0, -0.5, -1.2]])
lr4 = torch.tensor([[-1.5, -2.0, -1.2]])
kz = compute_kl_k3(lt4, lr4, torch.tensor([[1.0, 0.0, 1.0]]))
report("D mask 位置=0", kz[0, 1].item() == 0.0, f"k3[0,1] = {kz[0, 1].item():.1e}   k3[0,0] = {kz[0, 0].item():.6f}")
kb = compute_kl_k3(lt4, lr4, torch.ones((1, 3)))
report("D' 反证·mask全1", kb[0, 1].item() != 0.0, f"该位 → {kb[0, 1].item():.6f}")

# ---------- E Δ=0 锚点 ----------
same = torch.tensor([[-3.0, -1.0, -2.0]])
ke = compute_kl_k3(same, same, torch.ones((1, 3)))
report("E Δ=0 -> k3=0", float(ke.abs().max()) == 0.0, f"max|k3| = {ke.abs().max().item():.1e}")

# ---------- F 契约（[:,0] 用真实取值 = 0）----------
lt5 = torch.tensor([[0.0, -3.0, -1.2], [0.0, -0.7, -2.1]])
lr5 = torch.tensor([[0.0, -1.5, -1.2], [0.0, -2.3, -0.4]])
m5 = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
k5 = compute_kl_k3(lt5, lr5, m5)
report("F shape 同位", tuple(k5.shape) == tuple(lt5.shape), f"{tuple(k5.shape)} vs {tuple(lt5.shape)}")
report("F dtype 浮点", k5.is_floating_point(), f"{k5.dtype}")
report("F [:,0] == 0", bool((k5[:, 0] == 0).all()), f"k5[:,0] = {k5[:, 0].tolist()}")

print()
print(f"== {len(FAIL)} FAIL: {FAIL} ==" if FAIL else "== ALL PASS ==")
