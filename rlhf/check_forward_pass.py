"""P3 拼接闭环的检查：forward_pass（sample_rollout → logp → kl → r_sparse → reward）。

判据（每条都能失败）：
  A. 契约：返回 dict 里 7 个张量全 (B, S)；logp/kl/reward 浮点
  D'. 退化（关键对照）：actor==ref ⇒ kl.max()==0 且 reward==r_sparse
  —— 扰动 actor 一个参数、重跑 ⇒ kl.max()>0 且 reward!=r_sparse（链子真的活了）
     注意 rollout 走采样：actor 一变，即使 seed 相同 ids 也会变，所以对照的 r_sparse 要按新 ids 重算
  B. 梯度边界（在扰动后测）：reward 带 grad；reward.sum().backward() 后 actor 有非零 grad，
     而 ref / rm 参数零 grad（KL 项把梯度接给 actor，RM 标量必须是常数）
     ⚠️ 退化态不能测 B：Δ≡0 时 dk3/dΔ = e^Δ−1 = 0，梯度本就为 0（不是泄漏）
  C. 末位定位（扰动后）：r_sparse 非零列 == resp_mask 的 cumsum/argmax 末位；末位值 == score_pair
  D. kl 掩码（扰动后，非平凡）：resp_mask 之外严格 0，且 resp_mask 之内确有非零
  E. 不等长 prompt：attn_mask 行和必须不同，链不崩、形状一致
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from paths import ROOT  # noqa: E402

sys.path.insert(0, str(HERE.parent / "llama"))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE))

from llama_model import LLaMA  # noqa: E402
from rm import RM  # noqa: E402
from ppo_lm import forward_pass, compute_rewards  # noqa: E402

SFT_CKPT = ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt"
RM_CKPT = ROOT / "data" / "runs" / "rm_v3_S2000_lrh0.003_lrb0.003.pt"

MAXNEW = 20
BETA = 0.1


def load_model(ckpt):
    cfg = ckpt["config"]
    m = LLaMA(ckpt["vocab_size"], cfg["d_model"], cfg["n_layers"],
              cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


sft = torch.load(SFT_CKPT, map_location="cpu")
actor = load_model(sft)
ref = load_model(sft)
rm = RM(str(RM_CKPT))
rm.eval()

prompts = [torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22, 23])]


def run(prompts, max_new=MAXNEW, beta=BETA):
    # generate 默认 temperature=1.0 走 multinomial：不锁 seed，两次调用的 ids 就不是同一串
    torch.manual_seed(0)
    return forward_pass(actor, ref, rm, prompts, max_new, beta)


out = run(prompts)
ids, resp_mask = out["ids"], out["resp_mask"]
kl, reward = out["kl"], out["reward"]
B, S = ids.shape

# ---- A: 契约 ----
for name in ["ids", "attn_mask", "resp_mask", "logp_theta", "logp_ref", "kl", "reward"]:
    assert tuple(out[name].shape) == (B, S), f"A FAIL：{name} shape = {tuple(out[name].shape)} != {(B, S)}"
for name in ["logp_theta", "logp_ref", "kl", "reward"]:
    assert out[name].dtype.is_floating_point, f"A FAIL：{name} 必须是浮点"
print(f"A  contract PASS: 7 张量全 {(B, S)}；logp/kl/reward 浮点")

# ---- D': 退化对照（actor==ref）----
response_ids = [ids[i][resp_mask[i].bool()] for i in range(B)]
with torch.no_grad():
    r_sparse = compute_rewards(rm, prompts, response_ids, resp_mask)
assert kl.max().item() == 0.0, "D' FAIL：actor==ref 时 kl 应恒 0"
assert torch.equal(reward, r_sparse), "D' FAIL：kl==0 时 reward 应 == r_sparse"
print("D' degenerate: kl.max() = 0.0；reward == r_sparse")

# ---- 扰动 actor，同一 seed 重跑（Δ ≠ 0）----
with torch.no_grad():
    next(actor.parameters()).add_(0.01)  # 只动一个参数张量，打破 actor==ref
out2 = run(prompts)
ids2, resp_mask2 = out2["ids"], out2["resp_mask"]
kl2, reward2 = out2["kl"], out2["reward"]
response_ids2 = [ids2[i][resp_mask2[i].bool()] for i in range(B)]
with torch.no_grad():
    r_sparse2 = compute_rewards(rm, prompts, response_ids2, resp_mask2)
assert kl2.max().item() > 0, "D' FAIL：扰动 actor 后 kl 仍为 0"
assert not torch.equal(reward2, r_sparse2), "D' FAIL：扰动后 reward 仍 == r_sparse2（kl 没接进组装）"
print(f"D' perturbed PASS: kl.max() = {kl2.max().item():.3e} > 0；reward != r_sparse")

# ---- B: 梯度边界（扰动后测，Δ≠0 才有梯度）----
assert out2["logp_theta"].requires_grad, "B FAIL：logp_theta 必须带 grad（actor 路）"
assert reward2.requires_grad, "B FAIL：reward 必须带 grad"
assert not out2["logp_ref"].requires_grad, "B FAIL：logp_ref 不该带 grad"

actor.zero_grad(); ref.zero_grad(); rm.zero_grad()
reward2.sum().backward()
n_actor = sum(1 for p in actor.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_ref = sum(1 for p in ref.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_rm = sum(1 for p in rm.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
assert n_actor > 0, "B FAIL：backward 后 actor 一个参数都没拿到梯度"
assert n_ref == 0, f"B FAIL：梯度泄漏到 ref（{n_ref} 个参数非零）"
assert n_rm == 0, f"B FAIL：梯度泄漏到 rm（{n_rm} 个参数非零）"
print(f"B  grad PASS: actor {n_actor} 参数有 grad；泄漏 ref {n_ref} / rm {n_rm}")
actor.zero_grad(); ref.zero_grad(); rm.zero_grad()

# ---- C: 末位定位（扰动后）----
with torch.no_grad():
    score = rm.score_pair(prompts, response_ids2)
last = resp_mask2.cumsum(-1).argmax(-1)
nz = [r_sparse2[i].nonzero().flatten().tolist() for i in range(B)]
assert all(c == [int(l)] for c, l in zip(nz, last)), f"C FAIL：非零列 {nz} != 末位 {last.tolist()}"
d_score = (r_sparse2[torch.arange(B), last] - score).abs().max().item()
assert d_score == 0.0, f"C FAIL：末位值 != score_pair，max|d| = {d_score}"
print(f"C  terminal PASS: 非零列 {nz} == 末位 {last.tolist()}；末位值 == score_pair")

# ---- D: kl 掩码（扰动后，非平凡）----
off = kl2[resp_mask2 == 0].abs().max().item()
on = kl2[resp_mask2 == 1].abs().max().item()
assert off == 0.0, f"D FAIL：resp_mask 之外 kl 非零，max|v| = {off}"
assert on > 0, "D FAIL：resp_mask 之内 kl 全 0 —— 空检查"
print(f"D  kl-mask PASS: resp_mask 之外 max|v| = {off:.1e}，之内 max|v| = {on:.3e}")

# ---- E: 不等长 prompt ----
uneven = [torch.tensor([10]), torch.tensor([20, 21, 22, 23])]
outu = run(uneven)
su = outu["attn_mask"].sum(-1).tolist()
assert su[0] != su[1], f"E FAIL：不等长 prompt 的 attn_mask 行和相同 {su}"
assert tuple(outu["reward"].shape) == tuple(outu["ids"].shape), "E FAIL：reward / ids 形状不一致"
print(f"E  uneven PASS: attn_mask 行和 = {su}（不等）；reward == ids.shape {tuple(outu['ids'].shape)}")

print("\nALL PASS")
