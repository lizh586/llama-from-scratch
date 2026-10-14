"""P4-5 机制探针（只读 ckpt / 不训练 / 不改任何现有文件）。

上游读数（eval_kl + eval_hacking）：
  lr 1e-6 → actor 没动（drift 3e-4），实验空转；
  lr 1e-5 → policy 剧烈退化（EOS 0.652→0.000，L_mean 0.66→11.96），
            而 RM 分 r_φ 单调上升（-5.73→-2.38）、真值轴 hit 全程 0 ⇒ hacking 特征。

本脚本钉两件事：

  A 配对检验 —— r_φ 上升的**显著性**。
    eval_kl 打的是逐 prompt 的未配对 sem（≈1.0），所以 sft↔lr1e5 的差只有 1.4–2.3σ，
    说不死。这里用 common random numbers（同 EVAL_SEED + 同 torch 种子）
    做**逐 prompt 配对差** Δ_i = r_φ^(ckpt)(i) − r_φ^(sft)(i)，
    Δ 的 sem = std(Δ)/√n，这才是 CRN 换来的那个降方差检验。

  B RM 合成探针 —— r_φ 上升**归因到哪个轴**。
    对每条 SFT 生成 x（长 L），造两个等长的变体：
      ext  = x + filler          （filler = 下一条 prompt 的 SFT 生成，无自重复）
      rep  = x + x[:len(filler)] （把 x 自己的前缀接回去，自重复）
    两条长度相同 ⇒
      Δ_len = r_φ(ext) − r_φ(x)    只含「变长」效应
      Δ_rep = r_φ(rep) − r_φ(ext)  长度固定，只含「重复」效应
    哪个 Δ 显著，RM 就是被哪个轴喂出来的。

口径与 eval_kl / eval_hacking 完全同源：同一过滤、同一 EVAL_SEED、同一 CHUNK。
"""
import sys
from pathlib import Path

from paths import ROOT  # noqa: E402
LLM = Path(__file__).resolve().parents[1]
sys.path += [str(LLM / "llama"), str(LLM / "pipeline"), str(LLM / "sft")]

from llama_model import LLaMA
from data import vocab_size, T_DEFAULT, EOS_ID
import sft_data
import torch
from rm import RM
from ppo_lm import sample_rollout
import json
import random

ALPACA = ROOT / "data" / "sft" / "alpaca_sft.json"
SFT_CKPT = str(ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt")
RM_CKPT = str(ROOT / "data" / "runs" / "rm_v3_S2000_lrh0.003_lrb0.003.pt")

CKPTS = {
    "sft":         SFT_CKPT,
    "lr1e6":       str(ROOT / "data" / "runs" / "ppo_S200_B4_K4_lra1e-06_beta0.1_s0_actor.pt"),
    "lr1e5_s0":    str(ROOT / "data" / "runs" / "ppo_S200_B4_K4_lra1e-05_beta0.1_s0_actor.pt"),
    "lr1e5_s1":    str(ROOT / "data" / "runs" / "ppo_S200_B4_K4_lra1e-05_beta0.1_s1_actor.pt"),
}

N_EVAL = 256
CHUNK = 32
max_new_tokens = 64
EVAL_SEED = 12345
PAD_CAP = 32          # B 段变体里 response 的上限：prompt(≤190) + 2*32 = 254 ≤ 256，不撞 rm_batch 的 pad 断言
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_llama(path):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    model = LLaMA(vocab_size, cfg["d_model"], cfg["n_layers"],
                  cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def rollout_resps(actor, prompt_ids, L0=None):
    """跑一块 rollout，返回 (生成段 list of 1-D tensor, 本块 L0)。生成段不截 EOS —— 与 eval_kl 同口径。"""
    ids = sample_rollout(actor, prompt_ids, max_new_tokens)
    l0 = max(len(p) for p in prompt_ids)
    return [ids[j, l0:].clone() for j in range(ids.shape[0])], l0


def trunc_eos(gen):
    """截到第一个 EOS（含前的 token）；无 EOS 则原样。用于 B 段的"干净文本"。"""
    pos = (gen == EOS_ID).nonzero()
    return gen[:pos[0].item()] if len(pos) else gen


rows = json.load(open(ALPACA, "r", encoding="utf-8"))
rows = [e for e in rows if e["n"] <= T_DEFAULT - max_new_tokens]
random.seed(EVAL_SEED)
eval_rows = random.sample(rows, N_EVAL)
prompt_ids = [torch.tensor(sft_data.encode(sft_data.render(e["instruction"], e["input"], "")[0]))
              for e in eval_rows]

print(f"eval set: {N_EVAL} prompts, chunk {CHUNK}, max_new_tokens {max_new_tokens}, seed {EVAL_SEED}")

rm = RM(RM_CKPT).to(device)
rm.eval()

# ---- A 配对检验：逐 prompt 的 r_φ ----
scores = {}
for name, path in CKPTS.items():
    actor = build_llama(path)
    torch.manual_seed(EVAL_SEED)          # CRN：各 ckpt 共用同一条生成随机流
    acc = []
    for i in range(0, N_EVAL, CHUNK):
        chunk = prompt_ids[i:i + CHUNK]
        resps, _ = rollout_resps(actor, chunk)
        acc.append(rm.score_pair(chunk, resps).detach().cpu())
    scores[name] = torch.cat(acc)

base = scores["sft"]
print("\n== A  paired RM-score test (CRN, n=%d)  ==" % N_EVAL)
print(f"{'ckpt':10s} {'r_phi_mean':>11s} {'d_pair':>9s} {'d_sem':>8s} {'t':>7s} {'win%':>6s}")
for name, s in scores.items():
    d = s - base
    if name == "sft":
        print(f"{name:10s} {s.mean():>11.4f} {'--':>9s} {'--':>8s} {'--':>7s} {'--':>6s}")
        continue
    m = d.mean().item(); se = (d.std() / len(d) ** 0.5).item()
    t = m / se if se > 0 else float('nan')
    win = (d > 0).float().mean().item()
    print(f"{name:10s} {s.mean():>11.4f} {m:>+9.4f} {se:>8.4f} {t:>+7.2f} {win*100:>5.1f}%")

# ---- B RM 合成探针：长度 vs 重复 ----
actor_sft = build_llama(CKPTS["sft"])
torch.manual_seed(EVAL_SEED)
base_resps = []
for i in range(0, N_EVAL, CHUNK):
    resps, _ = rollout_resps(actor_sft, prompt_ids[i:i + CHUNK])
    base_resps += [trunc_eos(r)[:PAD_CAP] for r in resps]

idx = [i for i in range(N_EVAL) if len(base_resps[i]) and len(base_resps[(i + 1) % N_EVAL])]
probe_prompts = [prompt_ids[i] for i in idx]
x_list = [base_resps[i] for i in idx]
ext_list, rep_list = [], []
for i in idx:
    x = base_resps[i]
    f = base_resps[(i + 1) % N_EVAL][:PAD_CAP]
    f = f[:max(1, PAD_CAP - len(x))]      # 两条变体等长
    ext_list.append(torch.cat([x, f]))
    rep_list.append(torch.cat([x, x[:len(f)]]))

with torch.no_grad():
    s_x = torch.cat([rm.score_pair(probe_prompts[i:i + CHUNK], x_list[i:i + CHUNK]).detach().cpu()
                     for i in range(0, len(idx), CHUNK)])
    s_ext = torch.cat([rm.score_pair(probe_prompts[i:i + CHUNK], ext_list[i:i + CHUNK]).detach().cpu()
                       for i in range(0, len(idx), CHUNK)])
    s_rep = torch.cat([rm.score_pair(probe_prompts[i:i + CHUNK], rep_list[i:i + CHUNK]).detach().cpu()
                       for i in range(0, len(idx), CHUNK)])


def line(tag, a, b):
    d = b - a
    m = d.mean().item(); se = (d.std() / len(d) ** 0.5).item()
    t = m / se if se > 0 else float('nan')
    win = (d > 0).float().mean().item()
    return f"{tag:22s} {b.mean():>11.4f} {m:>+9.4f} {se:>8.4f} {t:>+7.2f} {win*100:>5.1f}%"


print(f"\n== B  RM synthetic probe (paired, n={len(idx)})  ==")
print(f"{'variant / contrast':22s} {'r_phi_mean':>11s} {'d':>9s} {'d_sem':>8s} {'t':>7s} {'win%':>6s}")
print(line("base (len L)", s_x, s_x))
print(line("ext  (len L+f)", s_x, s_ext))
print(line("rep  (len L+f)", s_x, s_rep))
print("-" * 60)
print(line("d_len = ext - base", s_x, s_ext))
print(line("d_rep = rep - ext", s_ext, s_rep))
