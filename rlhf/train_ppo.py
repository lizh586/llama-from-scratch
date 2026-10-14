import sys
from pathlib import Path
from paths import ROOT  # noqa: E402
LLM = Path(__file__).resolve().parents[1]
sys.path += [str(LLM/"llama"), str(LLM/"pipeline"), str(LLM/"sft")]
from llama_model import LLaMA
from data import stoi, itos, vocab_size, EOS_ID, T_DEFAULT
import sft_data
import torch
from rm import RM
from ppo_lm import Critic, forward_pass, ppo_step
import json
import random


ALPACA = ROOT / "data" / "sft" / "alpaca_sft.json"
B = 4

SFT_CKPT = str(ROOT/"data"/"runs"/"sft_lr1e-05_B32_T256_S1782.pt")
RM_CKPT = str(ROOT/"data"/"runs"/"rm_v3_S2000_lrh0.003_lrb0.003.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_llama(path):
    # SFT ckpt 存的是裸 LLaMA 键 ⇒ 直接 load_state_dict 进裸 LLaMA（actor/ref 都从这里起）。
    ckpt = torch.load(path)
    cfg = ckpt["config"]
    model = LLaMA(
        vocab_size, cfg["d_model"], cfg["n_layers"],
        cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model

# 四模型：actor 学、ref 供 KL 参照、rm 打常数分、critic 估价值。
# ref 是同一 ckpt 的第二个实例 —— 起点与 actor 逐位相同，之后不再更新。
actor  = build_llama(SFT_CKPT)
ref    = build_llama(SFT_CKPT)
rm     = RM(RM_CKPT).to(device)
critic = Critic(SFT_CKPT).to(device)   # backbone 从 SFT 起，价值头是随机初始化的

# ref / rm 必须冻：ref 一动 KL 基准就漂（参照系失效），rm 一动 reward 信号自指（自己给自己打分）
ref.requires_grad_(False)
rm.requires_grad_(False)

# 两个 optimizer 参数集不交叉。actor 微调（默认 1e-6，大了会把 SFT 学到的打崩）；
# critic 是随机初始化的头，用 1e-4 从头学价值函数。
# lr_actor 可命令行传（argv[3]）：P4-5 实测 1e-6 下 800 次更新只推得动 3e-4 参数位移 ⇒ 要扫更大 lr。
lr_actor = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
lr_critic = 1e-4
optim_actor  = torch.optim.Adam(actor.parameters(),  lr=lr_actor)
optim_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

with open(ALPACA, "r", encoding="utf-8") as f:
    rows = json.load(f)

max_new_tokens = 64
beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02   # P4-4 扫描用：命令行传 β
# T=256 是 prompt + response 的共用预算（不是各自上限）⇒ 池按 n ≤ 256 − max_new_tokens 过滤。
# 不过滤的话长 prompt + 生成段会超 256，rm_batch 的 pad 断言直接炸。max_new_tokens 一改，这行跟着改。
rows = [e for e in rows if e["n"] <= T_DEFAULT - max_new_tokens]
# 随机源有两处：prompt 批抽样走 python random，token 采样走 torch.multinomial（generate 内）。
# 只设 random.seed 不够 —— torch RNG 未播种时每次进程都不同，连"同 seed 复现"都做不到。
# 命令行传 seed（默认 0）：两次不同 seed = 两条独立轨迹，用来量噪声带。
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
K = 4          # ppo_step 的内层 epoch 数：一批 rollout 反复训 K 次
steps = 200
actor0 = [p.detach().clone() for p in actor.parameters()]   # 初始权重快照，用来量漂移（不受批间噪声影响）
history = []
for i in range(steps):
    batch = random.sample(rows, B)          # 每步重采一批 prompt ⇒ 纯 on-policy
    prompt_ids = []
    for e in batch:
        # render 第三参传空输出 = 只要 prompt；encode 出 list，包成 tensor（契约要求 list of tensor）
        s, n = sft_data.render(e["instruction"], e["input"], "")
        prompt_ids.append(torch.tensor(sft_data.encode(s)))

    out = forward_pass(actor, ref, rm, prompt_ids, max_new_tokens, beta)   # rollout + 打分，返回 7 键
    ppo_step(out, critic, actor, optim_critic, optim_actor, K=K)           # K 轮参数更新

    # 每步都算（不只打印那几步）：kl/rew 每步换 prompt 批，含批间噪声，要全长才看得出趋势
    resp = out["resp_mask"].bool()
    kl_m = out["kl"][resp].detach().mean().item()
    rew_m = out["reward"][resp].detach().mean().item()
    drift = max((p.detach() - q).abs().max().item()
            for p, q in zip(actor.parameters(), actor0))
    history.append({
        "step": i,
        "kl":   kl_m,
        "rew":  rew_m,
        "drift":drift
    })

    if i < 3 or i % 5 == 4:
        print(f"step {i:3d}  kl {kl_m:+.4e}  rew {rew_m:+.4f}  max|d|actor {drift:.3e}")

out_dir = ROOT / "data" / "runs"
out_dir.mkdir(parents=True, exist_ok=True)

tag = f"ppo_S{steps}_B{B}_K{K}_lra{lr_actor:g}_beta{beta:g}_s{seed}"   # 超参 + seed 进文件名 ⇒ 改超参/换 seed = 换文件，不覆写旧 run
actor_ckpt = {
    "config":torch.load(SFT_CKPT)["config"],   # build_llama 读完 config 就丢了，落盘要重取
    "vocab_size":vocab_size,
    # 裸 LLaMA 键（63）：build_llama 用 load_state_dict 读回；加前缀会 strict 抛 unexpected keys
    "state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
    "stoi":stoi,
    "itos":itos,
    "eos_id":EOS_ID,
}

torch.save(actor_ckpt, out_dir/f"{tag}_actor.pt")

critic_ckpt = {
    "config":critic.config,          # 已带 pooling="none" 溯源戳
    "vocab_size":vocab_size,
    # 存 critic.state_dict()（65 键 = model.* 63 + scalar_head.* 2），不是 critic.model.state_dict()：
    # 后者是裸 backbone，Critic() 读回时价值头会静默回到随机初始化（不报错，只有逐位比对能抓）
    "state_dict":{k:v.cpu() for k, v in critic.state_dict().items()},
    "stoi":stoi,
    "itos":itos,
    "eos_id":EOS_ID,
}

torch.save(critic_ckpt, out_dir/f"{tag}_critic.pt")

# run JSON 是过程记录（.pt 是结果）：P4-4 定 β 要读这里的 history 曲线
result = {
    "config":{
        "steps":steps, "B":B, "K": K,
        "max_new_tokens":max_new_tokens, "beta": beta,
        "lr_actor":lr_actor, "lr_critic": lr_critic,
        "seed": seed, "device":str(device)
    },
    "history":history,
}

with open(out_dir/f"{tag}.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
