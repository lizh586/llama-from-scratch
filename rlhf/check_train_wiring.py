"""check_train_wiring.py — train_ppo 接线层的静默失败点检查（只读；不跑训练循环）

不复验子函数（forward_pass / ppo_step / 各 loss 都有自己的 check）。
这里只验"把四个模型 + 两个 optimizer 摆到一起"才引入的东西：
实例身份、ckpt 没传反、冻结边界、optimizer 参数集、prompt 契约、退化态、step 副作用边界。

不 import train_ppo（那会连带跑一遍前向 + 一次 ppo_step），自带一份 setup 逐条对拍。
"""
from pathlib import Path
import sys, json, random, tempfile, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
LLM = HERE.parent
from paths import ROOT  # noqa: E402
sys.path += [str(LLM/"llama"), str(LLM/"pipeline"), str(LLM/"sft"), str(HERE)]

import torch
import sft_data
from llama_model import LLaMA
from data import itos, stoi, vocab_size, EOS_ID
from rm import RM
from ppo_lm import Critic, forward_pass, ppo_step, compute_logprobs, compute_kl_k3

FAIL = []


def report(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<32} {detail}")
    if not ok:
        FAIL.append(tag)


B = 4
SFT_CKPT = str(ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt")
RM_CKPT = str(ROOT / "data" / "runs" / "rm_v3_S2000_lrh0.003_lrb0.003.pt")
ALPACA = ROOT / "data" / "sft" / "alpaca_sft.json"
MAXNEW, BETA = 64, 0.02
PPO_TAG = "ppo_S20_B4_K4_lra1e-06_beta0.02"          # train_ppo.py 的超参拼出来的，改那边要同步
PPO_ACTOR = ROOT / "data" / "runs" / f"{PPO_TAG}_actor.pt"
PPO_CRITIC = ROOT / "data" / "runs" / f"{PPO_TAG}_critic.pt"
PPO_JSON = ROOT / "data" / "runs" / f"{PPO_TAG}.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEYS = ["ids", "attn_mask", "resp_mask", "logp_theta", "logp_ref", "kl", "reward"]


def build_llama(path):
    ckpt = torch.load(path)
    cfg = ckpt["config"]
    m = LLaMA(vocab_size, cfg["d_model"], cfg["n_layers"],
              cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    return m


def snap(m):
    return [p.detach().clone() for p in m.parameters()]


def changed(x, y):
    return any(not torch.equal(a, b) for a, b in zip(x, y))


def n_train(m):
    return sum(p.requires_grad for p in m.parameters()), sum(1 for _ in m.parameters())


def opt_ids(o):
    return {id(p) for g in o.param_groups for p in g["params"]}


actor = build_llama(SFT_CKPT)
ref = build_llama(SFT_CKPT)
rm = RM(RM_CKPT).to(device)
critic = Critic(SFT_CKPT).to(device)
ref.requires_grad_(False)
rm.requires_grad_(False)
optim_actor = torch.optim.Adam(actor.parameters(), lr=1e-6)
optim_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

# ---------- A 两个独立实例、权重逐位相同 ----------
sd_a, sd_r = actor.state_dict(), ref.state_dict()
report("A actor is ref", actor is not ref, f"actor is ref = {actor is ref}")
report("A actor≡ref 逐位相同",
       list(sd_a) == list(sd_r) and all(torch.equal(sd_a[k], sd_r[k]) for k in sd_a),
       f"比较了 {len(sd_a)} 张量")
with torch.no_grad():
    _p0 = next(ref.parameters())
    _p0_bak = _p0.detach().clone()          # 逐位备份：float32 下 p+ε−ε ≠ p，反向加减还原不回去
    _p0.add_(1e-4)
_perturbed = not all(torch.equal(sd_a[k], ref.state_dict()[k]) for k in sd_a)
with torch.no_grad():
    _p0.copy_(_p0_bak)
report("A' 反证·扰动 ref ⇒ 判据翻面", _perturbed, "改一个参数后逐位判据必须炸")

# ---------- B ckpt 没传反 ----------
sd_c = critic.model.state_dict()
rm_sd = rm.model.state_dict()
n_diff = sum(1 for k in sd_a if not torch.equal(rm_sd[k], sd_a[k]))
report("B critic backbone == actor",
       list(sd_c) == list(sd_a) and all(torch.equal(sd_c[k], sd_a[k]) for k in sd_a),
       "传成 RM_CKPT 这里会 False")
report("B rm backbone != actor", n_diff > 0, f"{n_diff}/{len(sd_a)} 张量不同")
report("B scalar_head 形状 (1,d_model)",
       tuple(critic.scalar_head.weight.shape) == (1, critic.config["d_model"]),
       f"{tuple(critic.scalar_head.weight.shape)}")
report("B pooling 戳 last/none",
       (rm.config["pooling"], critic.config["pooling"]) == ("last", "none"),
       f"{rm.config['pooling']!r} / {critic.config['pooling']!r}")

# ---------- C 冻结边界 ----------
for nm, m, want_zero in [("ref", ref, True), ("rm", rm, True),
                         ("actor", actor, False), ("critic", critic, False)]:
    nt, tot = n_train(m)
    ok = (nt == 0) if want_zero else (nt == tot)
    report(f"C {nm} 可训练参数", ok, f"{nt}/{tot}")
_lg = ref(torch.randint(0, vocab_size, (2, 4), device=device))
report("C ref 前向不带图", not _lg.requires_grad, f"req_grad = {_lg.requires_grad}")

# ---------- D optimizer 参数集不交叉 ----------
oa, oc = opt_ids(optim_actor), opt_ids(optim_critic)
a_ids = {id(p) for p in actor.parameters()}
c_ids = {id(p) for p in critic.parameters()}
fr_ids = {id(p) for p in ref.parameters()} | {id(p) for p in rm.parameters()}
report("D opt_actor 只含 actor", oa == a_ids, f"{len(oa)} vs {len(a_ids)}")
report("D opt_critic 只含 critic", oc == c_ids, f"{len(oc)} vs {len(c_ids)}")
report("D 两组零交集", not (oa & oc), f"交集 {len(oa & oc)}")
report("D ref/rm 未混入", len((oa | oc) & fr_ids) == 0, f"混入 {len((oa | oc) & fr_ids)} 个")
_oa_bad = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-6)
report("D' 反证·混合 opt ⇒ 判据翻面", opt_ids(_oa_bad) != a_ids,
       f"{len(opt_ids(_oa_bad))} vs {len(a_ids)}")

# ---------- E prompt 批次契约 ----------
with open(ALPACA, "r", encoding="utf-8") as f:
    rows = json.load(f)
random.seed(0)
batch = random.sample(rows, B)
prompt_ids, lens_ok = [], []
for e in batch:
    s, n = sft_data.render(e["instruction"], e["input"], "")
    prompt_ids.append(torch.tensor(sft_data.encode(s)))
    lens_ok.append(len(prompt_ids[-1]) == e["n"])
SUF = "### Response:\n"
report("E 条数 == B", len(prompt_ids) == B, f"{len(prompt_ids)}")
report("E 每条是 tensor", all(torch.is_tensor(p) for p in prompt_ids),
       "score_pair 要 .tolist()，裸 list 会 AttributeError")
report("E id 全在表内", all(0 <= int(i) < vocab_size for p in prompt_ids for i in p),
       f"vocab_size = {vocab_size}")
report("E 长度 == e['n'] 逐条对拍", all(lens_ok), f"{lens_ok}")
report("E 以响应标记收尾",
       all("".join(itos[int(i)] for i in p).endswith(SUF) for p in prompt_ids), repr(SUF))
report("E' 反证·裸 list 判据翻面",
       not all(torch.is_tensor(p) for p in [list(p.tolist()) for p in prompt_ids]),
       "回到 list of list 就会炸")

# ---------- F forward_pass 出口 ----------
out = forward_pass(actor, ref, rm, prompt_ids, MAXNEW, BETA)
B_, S_ = out["ids"].shape
L0 = max(len(p) for p in prompt_ids)
resp = out["resp_mask"].bool()
report("F 7 键名一致", list(out.keys()) == KEYS, str(list(out.keys())))
report("F 7 键同为 (B,S)", all(tuple(out[k].shape) == (B_, S_) for k in KEYS), f"({B_}, {S_})")
report("F dtype int64×3 + float32×4",
       all(out[k].dtype == torch.int64 for k in KEYS[:3])
       and all(out[k].is_floating_point() for k in KEYS[3:]), "ids/mask 整型，四个浮点")
report("F grad 边界",
       out["logp_theta"].requires_grad and not out["logp_ref"].requires_grad
       and out["kl"].requires_grad and out["reward"].requires_grad,
       "θ/kl/reward 带图，ref 不带")
report("F resp_mask 语义", bool((out["resp_mask"][:, :L0] == 0).all())
       and bool((out["resp_mask"][:, L0:] == 1).all()), f"前 L0={L0} 列全 0")
report("F attn_mask 行和 == len(prompt)",
       all(int(out["attn_mask"][i, :L0].sum()) == len(prompt_ids[i]) for i in range(B_)), "")

# ---------- G 退化态 + 反证 ----------
report("G actor≡ref ⇒ kl ≡ 0",
       torch.equal(out["logp_theta"].detach(), out["logp_ref"])
       and float(out["kl"][resp].detach().abs().max()) == 0.0, "Δ=0 ⇒ k3=0")
actor.zero_grad()
out["reward"].sum().backward()
g0 = sum(p.grad.abs().sum().item() for p in actor.parameters() if p.grad is not None)
n_none = sum(1 for p in actor.parameters() if p.grad is None)
n_tot = sum(1 for _ in actor.parameters())
report("G step-0 |grad| == 0", g0 == 0.0, f"{g0:.3e}")
report("G 图是通的(grad 非 None)", n_none == 0, f"None 个数 = {n_none}/{n_tot}")
actor.zero_grad()
kl2 = compute_kl_k3(compute_logprobs(actor, out["ids"], out["attn_mask"]),
                    out["logp_ref"] + 0.1, out["resp_mask"])
kl2.sum().backward()
g1 = sum(p.grad.abs().sum().item() for p in actor.parameters() if p.grad is not None)
report("G' 反证·Δ=0.1 ⇒ grad > 0", g1 > 0, f"|grad| = {g1:.3e}")
actor.zero_grad()

# ---------- H ppo_step 副作用边界 ----------
MODELS = [("actor", actor), ("critic", critic), ("ref", ref), ("rm", rm)]
s0 = {n: snap(m) for n, m in MODELS}
keys_before = list(out.keys())
ppo_step(out, critic, actor, optim_critic, optim_actor, K=4)
s1 = {n: snap(m) for n, m in MODELS}
report("H actor 被更新", changed(s0["actor"], s1["actor"]), "应 True")
report("H critic 被更新", changed(s0["critic"], s1["critic"]), "应 True")
report("H ref 未动", not changed(s0["ref"], s1["ref"]), "应 False")
report("H rm 未动", not changed(s0["rm"], s1["rm"]), "应 False")
report("H out 未被 mutate", list(out.keys()) == keys_before, "ppo_step 不该动入参")
report("H 无 NaN/Inf",
       all(torch.isfinite(p).all() for m in (actor, critic) for p in m.parameters()), "")

a3 = build_llama(SFT_CKPT)
c3 = Critic(SFT_CKPT).to(device)
s_c3 = snap(c3)
ppo_step(out, c3, a3, torch.optim.Adam(c3.parameters(), lr=1e-4),
         torch.optim.Adam(a3.parameters(), lr=1e-6), K=2, c_v=0.0)
report("H' 反证·c_v=0 ⇒ critic 冻结", not changed(s_c3, snap(c3)),
       "critic 未被 actor 那条路污染")

# ---------- I 落盘 round-trip（读 train_ppo 产出的三个文件）----------
report("I 三个文件都存在",
       PPO_ACTOR.exists() and PPO_CRITIC.exists() and PPO_JSON.exists(), PPO_TAG)

ac, cc = torch.load(PPO_ACTOR), torch.load(PPO_CRITIC)
a_keys, c_keys = list(ac["state_dict"]), list(cc["state_dict"])

# 键空间：actor 裸 63 键（无前缀）；critic 65 键 = model.* 63（56 参数 + 7 buffer）+ scalar_head.* 2
report("I actor 键全裸", len(a_keys) == 63
       and not any(k.startswith(("model.", "scalar_head.")) for k in a_keys), f"{len(a_keys)} 键")
report("I critic 键带前缀", len(c_keys) == 65
       and sum(k.startswith("model.") for k in c_keys) == 63
       and {"scalar_head.weight", "scalar_head.bias"} <= set(c_keys),
       f"{len(c_keys)} 键 = model.* {sum(k.startswith('model.') for k in c_keys)} + scalar_head.* 2")

# round-trip：读回不抛 = 键严格对齐（load_state_dict strict=True）；再逐位比对
aa = build_llama(str(PPO_ACTOR))
report("I actor round-trip 逐位相同",
       all(torch.equal(aa.state_dict()[k].cpu(), ac["state_dict"][k]) for k in ac["state_dict"]),
       "build_llama 读回")
ca = Critic(str(PPO_CRITIC)).to(device)
report("I critic round-trip 逐位相同（含价值头）",
       all(torch.equal(ca.state_dict()[k].cpu(), cc["state_dict"][k]) for k in cc["state_dict"]),
       "Critic() 整体加载")

# 值不是字符串：今天修的就是这个（"itos":"itos"）
report("I stoi/itos/eos_id 是值",
       all(isinstance(d["stoi"], dict) and isinstance(d["itos"], dict)
           and isinstance(d["eos_id"], int) for d in (ac, cc))
       and ac["stoi"] == stoi and ac["itos"] == itos and ac["eos_id"] == EOS_ID,
       "抓字符串字面量 bug")

# run JSON 结构
with open(PPO_JSON, encoding="utf-8") as f:
    run = json.load(f)
report("I run JSON history 20 条 × 4 键",
       len(run["history"]) == 20
       and all(set(r) == {"step", "kl", "rew", "drift"} for r in run["history"]),
       f"{len(run['history'])} 条，键 {sorted(run['history'][0])}")
report("I run JSON config 9 字段",
       set(run["config"]) == {"steps", "B", "K", "max_new_tokens", "beta",
                              "lr_actor", "lr_critic", "seed", "device"}, str(sorted(run["config"])))

# 反证 · critic 存裸 backbone：Critic() 不抛，但价值头静默回到随机初始化
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
    _tmp = tf.name
torch.save({"config": cc["config"], "vocab_size": vocab_size,
            "state_dict": {k[len("model."):]: v for k, v in cc["state_dict"].items()
                           if k.startswith("model.")}}, _tmp)
cb = Critic(_tmp).to(device)
report("I' 反证·critic 存裸 backbone ⇒ 不抛但头变随机",
       not torch.equal(cb.scalar_head.weight.detach().cpu(), cc["state_dict"]["scalar_head.weight"]),
       "静默失效：只有逐位比对能抓到")

# 反证 · actor 加前缀：口径错会响（区别于 critic 的静默）
torch.save({"config": ac["config"], "vocab_size": vocab_size,
            "state_dict": {f"model.{k}": v for k, v in ac["state_dict"].items()}}, _tmp)
_raised = False
try:
    build_llama(_tmp)
except RuntimeError:
    _raised = True
report("I' 反证·actor 加前缀 ⇒ 抛", _raised, "load_state_dict strict 报 unexpected keys")
os.remove(_tmp)

print()
print(f"== {len(FAIL)} FAIL: {FAIL} ==" if FAIL else "== ALL PASS ==")
