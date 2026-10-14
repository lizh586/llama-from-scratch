import sys
from pathlib import Path
from paths import ROOT  # noqa: E402
LLM = Path(__file__).resolve().parents[1]
sys.path += [str(LLM/"llama"), str(LLM/"pipeline"), str(LLM/"sft")]
from llama_model import LLaMA
from data import vocab_size, T_DEFAULT
import sft_data
import torch
from rm import RM
from ppo_lm import compute_logprobs, compute_kl_k3, sample_rollout
import json
import random


ALPACA   = ROOT / "data" / "sft" / "alpaca_sft.json"
SFT_CKPT = str(ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt")
RM_CKPT  = str(ROOT / "data" / "runs" / "rm_v3_S2000_lrh0.003_lrb0.003.pt")
PPO_TAG  = "ppo_S200_B4_K4_lra1e-06_beta0.02"
# 命令行可传 name=path 覆盖（P4-4 pilot 的四个 β ckpt 走这条）；无参回退到 10/8 的 sft/s0/s1。
if len(sys.argv) > 1:
    ACTORS = dict(a.split("=", 1) for a in sys.argv[1:])
else:
    ACTORS = {
        "sft": SFT_CKPT,
        "s0":  str(ROOT / "data" / "runs" / f"{PPO_TAG}_s0_actor.pt"),
        "s1":  str(ROOT / "data" / "runs" / f"{PPO_TAG}_s1_actor.pt"),
    }

N_EVAL         = 256    # 固定评测 prompt 数（训练时每步才 4 条 → 这里是它的 64 倍）
CHUNK          = 32     # 分块 rollout，控显存
max_new_tokens = 64
EVAL_SEED      = 12345  # 评测集 + 生成随机数都与训练 seed 无关，且三个 ckpt 共用（common random numbers）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_llama(path):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    model = LLaMA(vocab_size, cfg["d_model"], cfg["n_layers"],
                  cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


# 与 forward_pass 同构的 mask 构造 + kl 计算，只是把 batch 放大、prompt 固定。
def eval_chunk(actor, ref, prompt_ids):
    ids = sample_rollout(actor, prompt_ids, max_new_tokens)
    L0 = max(len(p) for p in prompt_ids)
    B, S = ids.shape
    attn_mask = torch.zeros((B, S), dtype=torch.long, device=ids.device)
    resp_mask = torch.zeros((B, S), dtype=torch.long, device=ids.device)
    for i, p in enumerate(prompt_ids):
        attn_mask[i, L0 - len(p):] = 1
    resp_mask[:, L0:] = 1

    logp_theta = compute_logprobs(actor, ids, attn_mask=attn_mask)
    with torch.no_grad():
        logp_ref = compute_logprobs(ref, ids, attn_mask=attn_mask)
    kl = compute_kl_k3(logp_theta, logp_ref, resp_mask)        # (B,S)，resp 外已置 0
    lens = resp_mask.sum(-1)                                   # (B,)
    kl_per = kl.detach().sum(-1) / lens                        # 每条 prompt 的逐 token 平均 KL

    response_ids = [ids[i][resp_mask[i].bool()] for i in range(B)]
    rew_per = rm.score_pair(prompt_ids, response_ids).detach()  # (B,) 序列级 r_φ
    return kl_per.cpu(), rew_per.cpu()


def mean_sem(x):
    return x.mean().item(), (x.std() / len(x) ** 0.5).item()


rows = json.load(open(ALPACA, "r", encoding="utf-8"))
rows = [e for e in rows if e["n"] <= T_DEFAULT - max_new_tokens]
random.seed(EVAL_SEED)
eval_rows = random.sample(rows, N_EVAL)
prompt_ids = [torch.tensor(sft_data.encode(sft_data.render(e["instruction"], e["input"], "")[0]))
              for e in eval_rows]
print(f"eval set: {N_EVAL} prompts, chunk {CHUNK}, max_new_tokens {max_new_tokens}, "
      f"seed {EVAL_SEED}  (max n = {max(e['n'] for e in eval_rows)})")

ref = build_llama(SFT_CKPT)
ref.requires_grad_(False)
ref.eval()
rm = RM(RM_CKPT).to(device)
rm.eval()

print(f"\n{'ckpt':5s}  {'KL mean':>12s}  {'KL sem':>10s}  {'KL/N':>8s}  {'rew mean':>10s}  {'rew sem':>9s}")
for name, path in ACTORS.items():
    actor = build_llama(path)
    actor.eval()
    torch.manual_seed(EVAL_SEED)     # 三个 ckpt 共用同一条生成随机流 ⇒ 差异里少一项采样噪声
    kl_all, rew_all = [], []
    for i in range(0, N_EVAL, CHUNK):
        k, r = eval_chunk(actor, ref, prompt_ids[i:i + CHUNK])
        kl_all.append(k); rew_all.append(r)
    kl_all = torch.cat(kl_all); rew_all = torch.cat(rew_all)
    m, s = mean_sem(kl_all)
    rm_, rs = mean_sem(rew_all)
    print(f"{name:5s}  {m:>12.6f}  {s:>10.2e}  {len(kl_all):>8d}  {rm_:>10.6f}  {rs:>9.2e}")
