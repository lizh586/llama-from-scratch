"""check_critic.py — Critic.forward 的静默失败点检查（只读；不改 ppo_lm.py / 不训练）

判据来源：P3 ④ —— Critic 复用 RM backbone，价值头逐位出值 -> (B, S)。

关键静默失败点（错了不会报错，只会把值算错）：
  1. 把 RM 的末位池化照抄过来 -> 形状 (B,)，P4 相加时才会炸（或广播错）
  2. attn_mask 没透传进 backbone -> 左 padding 参与 attention，形状全对但值错
  3. head 被 detach / backbone 冻住 -> 形状对、前向对，P4 反传不到

注意：head 是随机初始化的，输出数值本身无意义，这里只查结构/语义/梯度。
"""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path += [str(Path(__file__).resolve().parent)]
import torch
from ppo_lm import Critic
from data import PAD_ID

FAIL = []


def report(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<26} {detail}")
    if not ok:
        FAIL.append(tag)


from paths import ROOT  # noqa: E402
CKPT = ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt"

torch.manual_seed(0)
critic = Critic(str(CKPT)).to("cpu").eval()
D = critic.config["d_model"]
V = critic.model.embedding.num_embeddings
print(f"ckpt = {CKPT.name}")
print(f"d_model = {D}   vocab = {V}   PAD_ID = {PAD_ID}")
print()

ids = torch.tensor([[5, 7, 11, 13, 17, 19],
                    [23, 29, 31, 37, 41, 43]])
mask = torch.ones_like(ids)
out = critic(ids, mask)

# ---------- A 形状契约 ----------
report("A shape == (B,S)", tuple(out.shape) == tuple(ids.shape),
       f"{tuple(out.shape)} vs {tuple(ids.shape)}")

h = critic.model.get_backbone(ids, mask)
last = critic.scalar_head(h[:, -1, :]).squeeze(-1)   # RM 式末位池化
report("A' 反证·末位池化", tuple(last.shape) != tuple(ids.shape),
       f"末位池化 -> {tuple(last.shape)}   (不等于 {(len(ids), ids.shape[1])} => 检查有鉴别力)")

# ---------- B 逐位出值（不是把一个值广播到所有位置）----------
W = critic.scalar_head.weight.detach()
b = critic.scalar_head.bias.detach()
manual = (h.detach() @ W.t() + b).squeeze(-1)       # 独立于 nn.Linear 的重算
dB = (out - manual).abs().max().item()
report("B out == backbone@W^T + b", dB <= 1e-6, f"max|Δ| = {dB:.2e}")

bad_b = manual[:, -1:].expand_as(manual)            # 若照抄 RM：整个序列用末位值
spread = (bad_b - manual).abs().max().item()
report("B' 反证·末位广播", spread > 1e-3,
       f"与逐位结果 max|Δ| = {spread:.3e}   (应 >0 => 逐位检查抓得住)")

# ---------- C mask 透传（左 padding 批量 == 单条）----------
# 只比有 padding 的那一行：无 padding 的行 mask 本来就不起作用，比了是空检查。
p1 = [5, 7, 11, 13, 17]
p2 = [23, 29, 31]                                   # 左 pad 2 位
L0 = len(p1)
ids_pad = torch.full((2, L0), PAD_ID, dtype=torch.long)
mask_pad = torch.zeros((2, L0), dtype=torch.long)
ids_pad[0, L0 - len(p1):] = torch.tensor(p1)
mask_pad[0, L0 - len(p1):] = 1
ids_pad[1, L0 - len(p2):] = torch.tensor(p2)
mask_pad[1, L0 - len(p2):] = 1

out_pad = critic(ids_pad, mask_pad)
out_s2 = critic(torch.tensor([p2]), torch.ones(1, len(p2), dtype=torch.long))[0]
dC = (out_pad[1, L0 - len(p2):] - out_s2).abs().max().item()
report("C 批量(左pad) == 单条", dC <= 1e-5,
       f"pad 行 max|Δ| = {dC:.2e}   (比对的是长度 {len(p2)} 的 pad 行)")

out_nomask = critic(ids_pad, None)                  # padding 也参与 attention
dCb = (out_nomask[1, L0 - len(p2):] - out_s2).abs().max().item()
report("C' 反证·不传 mask", dCb > 1e-4,
       f"max|Δ| = {dCb:.2e}   (应 >0 => mask 确实在起作用)")

# ---------- D 梯度双向可达 ----------
critic.zero_grad()
critic(ids, mask).sum().backward()
g_emb = critic.model.embedding.weight.grad
g_head = critic.scalar_head.weight.grad
ok_d = (g_emb is not None and torch.isfinite(g_emb).all() and g_emb.abs().sum() > 0
        and g_head is not None and torch.isfinite(g_head).all() and g_head.abs().sum() > 0)
report("D 梯度到 backbone+head", bool(ok_d),
       f"emb |g|sum = {None if g_emb is None else g_emb.abs().sum().item():.3e}   "
       f"head |g|sum = {None if g_head is None else g_head.abs().sum().item():.3e}")

critic.zero_grad()
with torch.no_grad():
    h_frozen = critic.model.get_backbone(ids, mask)
critic.scalar_head(h_frozen).sum().backward()
report("D' 反证·backbone 冻住", critic.model.embedding.weight.grad is None,
       f"emb grad = {critic.model.embedding.weight.grad}   (None => 检查抓得住冻结)")

# ---------- E bias=True 生效（绝对标度，偏置不可消）----------
with torch.no_grad():
    critic.scalar_head.weight.zero_()
out_e = critic(ids, mask)
bias_val = critic.scalar_head.bias.item()
const_ok = (out_e - out_e[:, :1]).abs().max().item() == 0.0
match_ok = abs(out_e[0, 0].item() - bias_val) <= 1e-6
report("E 权重清零 -> out ≡ bias", const_ok and match_ok,
       f"out = {out_e[0, 0].item():.6f}   bias = {bias_val:.6f}   跨位差 = {(out_e - out_e[:, :1]).abs().max().item():.1e}")
report("E' 反证·bias=False", abs(bias_val) > 1e-4 and abs(0.0 - bias_val) > 1e-4,
       f"bias=False 时输出恒 0，与实测 {bias_val:.6f} 差 {abs(bias_val):.3e}")

# ---------- F 值域 ----------
report("F 有限浮点", bool(torch.isfinite(out).all()) and out.is_floating_point(),
       f"dtype = {out.dtype}   finite = {bool(torch.isfinite(out).all())}")

print()
print(f"== {len(FAIL)} FAIL: {FAIL} ==" if FAIL else "== ALL PASS ==")
