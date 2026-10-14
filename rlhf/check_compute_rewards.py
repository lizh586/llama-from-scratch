"""P3 出口 RM scoring 的检查：compute_rewards。

判据（每条都能失败）：
  A. 契约：shape == mask.shape；非末位（含 prompt 位 + gen 后尾部 pad）严格 0
  B. 手算对照：末位列 == 从构造直接算出的 [5,4,7]（不等长，写死 S-1 会给错列）；
     末位值 == rm.score_pair 的原始输出，逐行
  B'. 反证：把 last 写死成 S-1（每行都写最后一列），读数必须变
  C. 定位跟 mask 走：改一位 mask → 非零列随之移动
  D. device：GPU 上跑不报 mismatch，读数与 CPU 一致
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from paths import ROOT  # noqa: E402

sys.path.insert(0, str(HERE.parent / "llama"))
sys.path.insert(0, str(HERE.parent / "pipeline"))
sys.path.insert(0, str(HERE))

from data import PAD_ID  # noqa: E402
from ppo_lm import compute_rewards  # noqa: E402
from rm import RM  # noqa: E402

CKPT = ROOT / "data" / "runs" / "rm_v3_S2000_lrh0.003_lrb0.003.pt"

# ---- 构造 rollout 风格 batch：prompt 左 pad 到 L0，gen 从 L0 起，gen 后补 pad ----
# PAD_ID == EOS_ID == 96 ⇒ gen 段内的 pad 与真实 EOS 无法用 id 区分，只能靠 mask 标出。
prompts = [torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22, 23]), torch.tensor([30])]
gens = [torch.tensor([40, 41]), torch.tensor([50]), torch.tensor([60, 61, 62, 63])]
B = len(prompts)
L0 = max(len(p) for p in prompts)  # 4
S = L0 + max(len(g) for g in gens)  # 8

x = torch.full((B, S), PAD_ID, dtype=torch.long)
resp_mask = torch.zeros((B, S), dtype=torch.long)
for i, (p, g) in enumerate(zip(prompts, gens)):
    x[i, L0 - len(p):L0] = p
    x[i, L0:L0 + len(g)] = g
    resp_mask[i, L0:L0 + len(g)] = 1

expected_last = torch.tensor([L0 + len(g) - 1 for g in gens])  # [5, 4, 7]
print(f"  batch: B={B} L0={L0} S={S}  expected_last={expected_last.tolist()}")

rm = RM(str(CKPT))
rm.eval()

with torch.no_grad():
    reward = compute_rewards(rm, prompts, gens, resp_mask)
    score = rm.score_pair(prompts, gens)

rows = torch.arange(B)

# ---- A: 契约 ----
assert tuple(reward.shape) == (B, S), f"shape 应为 {(B, S)}，实际 {tuple(reward.shape)}"
keep = torch.zeros_like(reward, dtype=torch.bool)
keep[rows, expected_last] = True
off = reward[~keep].abs().max().item()
assert off == 0.0, f"判据 A FAIL：非末位有 {off} 个非零值（prompt 位 / 尾部 pad 未清零）"
print(f"A  contract PASS: shape {tuple(reward.shape)} / 非末位 max|v| = {off:.1e}")

# ---- B: 手算对照 ----
last = resp_mask.int().cumsum(-1).argmax(-1)
d_last = (last != expected_last).sum().item()
assert d_last == 0, f"判据 B FAIL：末位列 {last.tolist()} != 手算 {expected_last.tolist()}"
d_score = (reward[rows, expected_last] - score).abs().max().item()
assert d_score == 0.0, f"判据 B FAIL：末位值 != score_pair 输出，max|d| = {d_score}"
print(f"B  manual PASS: 末位列 {last.tolist()} == 手算 / 末位值 max|d| = {d_score:.1e}")

# ---- B': 反证 —— 每行都写最后一列（等价写死 S-1）----
wrong = torch.zeros_like(reward)
wrong[:, S - 1] = score
d_wrong = (reward - wrong).abs().max().item()
assert d_wrong > 0, "反证 FAIL：不等长下写成 S-1 居然无差别 —— B 是空检查"
print(f"B' control  write-S-1 max|d| = {d_wrong:.2e}")

# ---- C: 定位跟 mask 走 —— 第 0 行 response 缩到 1 个 token，非零列必须移到 L0 ----
mask2 = resp_mask.clone()
mask2[0, L0 + 1] = 0
with torch.no_grad():
    r2 = compute_rewards(rm, prompts, gens, mask2)
nz2 = r2[0].nonzero().flatten().tolist()
assert nz2 == [L0], f"判据 C FAIL：改 mask 后非零列 = {nz2}，应为 [{L0}]"
assert r2[0, L0].item() == score[0].item(), "判据 C FAIL：移动后的末位值 != score"
print(f"C  mask-driven PASS: 非零列 {nz2} == [{L0}]")

# ---- D: device ----
if torch.cuda.is_available():
    rm_gpu = RM(str(CKPT)).to("cuda")
    rm_gpu.eval()
    with torch.no_grad():
        r_gpu = compute_rewards(rm_gpu, prompts, gens, resp_mask.to("cuda"))
    d_dev = (r_gpu.cpu() - reward).abs().max().item()
    assert d_dev < 1e-5, f"判据 D FAIL：GPU/CPU 不一致 max|d| = {d_dev}"
    print(f"D  device PASS: cuda 无 mismatch, cuda-vs-cpu max|d| = {d_dev:.2e}")
else:
    print("D  device SKIP: 无 CUDA")

print("\nALL PASS")
