"""check_actor_critic_loss.py — P4-2 actor / critic loss 静默失败点检查（只读；不碰 ckpt / ppo_lm 本体）"""
from pathlib import Path
import sys, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path += [str(Path(__file__).resolve().parent)]
import torch
from ppo_lm import actor_loss, critic_loss

FAIL = []
EPS = 0.2


def report(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<28} {detail}")
    if not ok:
        FAIL.append(tag)


def hand_actor(lt, lo, adv, mask, eps):
    """独立实现：python float + math.exp；返回 -L^CLIP"""
    tot, n = 0.0, 0
    for b in range(len(lt)):
        for s in range(len(lt[0])):
            if mask[b][s] != 1:
                continue
            ro = math.exp(lt[b][s] - lo[b][s])
            rc = min(max(ro, 1 - eps), 1 + eps)
            tot += min(ro * adv[b][s], rc * adv[b][s])
            n += 1
    return -(tot / n)


def hand_critic(v, vo, adv, mask):
    """独立实现：masked MSE"""
    tot, n = 0.0, 0
    for b in range(len(v)):
        for s in range(len(v[0])):
            if mask[b][s] != 1:
                continue
            tgt = adv[b][s] + vo[b][s]
            tot += (v[b][s] - tgt) ** 2
            n += 1
    return tot / n


lt = torch.tensor([[-1.0, -0.5, -2.0, -0.3], [-0.2, -1.5, -0.9, -1.1]])
lo = torch.tensor([[-1.5, -0.5, -1.5, -0.8], [-0.7, -1.5, -1.4, -1.1]])
# adv[0,3] 取 +0.5 而非负值：唯有 ρ>1+ε 且 A>0 时 min 才取截断项，clip 才"咬得上"，
# 否则「无 clip」反证恒等于参考值（首版数据就踩了这个，反证炸不出 = 检查形同虚设）。
adv = torch.tensor([[1.0, -1.0, 2.0, 0.5], [0.5, 1.0, -2.0, -1.0]])
mask = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])

# ---------- A actor 手算对照 ----------
a_out = actor_loss(lt, lo, adv, mask, EPS)
a_ref = hand_actor(lt.tolist(), lo.tolist(), adv.tolist(), mask.tolist(), EPS)
dA = abs(a_out.item() - a_ref)
report("A actor 手算对照", dA <= 1e-6, f"|Δ| = {dA:.2e}   out={a_out.item():.6f} hand={a_ref:.6f}")

# ---------- A' actor 四种坏法必须对不上 ----------
ro = torch.exp(lt - lo)
unc = ro * adv
clp = torch.clip(ro, 1 - EPS, 1 + EPS) * adv
bad_a = {
    "少负号": torch.min(unc, clp)[mask.bool()].mean(),
    "无 clip": -(unc)[mask.bool()].mean(),
    "取 max": -torch.max(unc, clp)[mask.bool()].mean(),
    "分母整块": -torch.min(unc, clp)[mask.bool()].sum() / (lt.shape[0] * lt.shape[1]),
}
for name, b in bad_a.items():
    db = abs(b.item() - a_ref)
    report(f"A' 反证·{name}", db > 1e-3, f"|Δ vs 手算| = {db:.3e}")

# ---------- B clip 生效（两个方向）----------
# ρ>1+ε 且 A>0 → min 取截断项；ρ>1+ε 且 A<0 → min 取未截断项
lt2, lo2 = torch.tensor([[0.0, 0.0]]), torch.tensor([[-1.0, -1.0]])   # ρ = e ≈ 2.72 > 1.2
m2 = torch.ones((1, 2))
adv2 = torch.tensor([[1.0, -1.0]])
o2 = actor_loss(lt2, lo2, adv2, m2, EPS)
rho = math.exp(1.0)
hand2 = -(min(rho * 1.0, 1.2 * 1.0) + min(rho * -1.0, 1.2 * -1.0)) / 2
report("B clip 生效(A>0取截断)", abs(o2.item() - hand2) <= 1e-6, f"out={o2.item():.6f} hand={hand2:.6f}")
big = -(rho * 1.0 + rho * -1.0) / 2   # 不 clip 的话
report("B' 反证·不 clip", abs(big - hand2) > 1e-3, f"不clip={big:.6f} 差 {abs(big - hand2):.3e}")

# ---------- C critic 手算对照 ----------
v = torch.tensor([[0.5, -0.3, 1.2, 0.0], [0.1, 0.8, -0.6, 0.4]])
vo = torch.tensor([[0.4, -0.2, 1.0, 0.1], [0.0, 0.7, -0.5, 0.3]])
c_out = critic_loss(v, vo, adv, mask)
c_ref = hand_critic(v.tolist(), vo.tolist(), adv.tolist(), mask.tolist())
dC = abs(c_out.item() - c_ref)
report("C critic 手算对照", dC <= 1e-6, f"|Δ| = {dC:.2e}   out={c_out.item():.6f} hand={c_ref:.6f}")

# ---------- C' critic 三种坏法必须对不上 ----------
tgt = adv + vo
bad_c = {
    "漏 .sum()": (mask * (v - tgt) ** 2).sum() / mask.sum() / 16,  # 形状/口径都不对
    "分母整块": ((mask * (v - tgt) ** 2).sum() / (v.shape[0] * v.shape[1])),
    "target 缺 V_old": ((mask * (v - adv) ** 2).sum() / mask.sum()),
}
for name, b in bad_c.items():
    db = abs(b.item() - c_ref)
    report(f"C' 反证·{name}", db > 1e-3, f"|Δ vs 手算| = {db:.3e}")

# ---------- D 输出形状 = 标量 ----------
report("D actor 输出标量", a_out.dim() == 0, f"dim={a_out.dim()} shape={tuple(a_out.shape)}")
report("D critic 输出标量", c_out.dim() == 0, f"dim={c_out.dim()} shape={tuple(c_out.shape)}")

# ---------- E detach 契约：调用方传快照 ----------
lt_g = lt.clone().requires_grad_(True)
lo_snap = lo.clone()                       # requires_grad=False = 快照
actor_loss(lt_g, lo_snap, adv, mask, EPS).backward()
report("E actor 梯度进 θ", lt_g.grad is not None and float(lt_g.grad.abs().sum()) > 0,
       f"|grad θ| = {float(lt_g.grad.abs().sum()):.3e}")
report("E actor 不进 snapshot", lo_snap.grad is None, f"logp_old.grad = {lo_snap.grad}")

lt_g2 = lt.clone().requires_grad_(True)
lo_leak = lo.clone().requires_grad_(True)   # 忘了快照
actor_loss(lt_g2, lo_leak, adv, mask, EPS).backward()
report("E' 反证·不传快照→泄漏", lo_leak.grad is not None and float(lo_leak.grad.abs().sum()) > 0,
       f"logp_old.grad = {float(lo_leak.grad.abs().sum()):.3e}  (应非 0)")

v_g = v.clone().requires_grad_(True)
vo_snap = vo.clone()
critic_loss(v_g, vo_snap, adv, mask).backward()
report("E critic 梯度进 V_φ", v_g.grad is not None and float(v_g.grad.abs().sum()) > 0,
       f"|grad V_φ| = {float(v_g.grad.abs().sum()):.3e}")
report("E critic 不进 snapshot", vo_snap.grad is None, f"V_old.grad = {vo_snap.grad}")

print()
print(f"== {len(FAIL)} FAIL: {FAIL} ==" if FAIL else "== ALL PASS ==")
