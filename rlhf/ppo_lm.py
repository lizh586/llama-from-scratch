from pathlib import Path
import sys
LLM = Path(__file__).resolve().parents[1]
sys.path += [str(LLM/"pipeline"), str(LLM/"llama")]
from llama_model import LLaMA
from data import PAD_ID, EOS_ID
import torch
import torch.nn.functional as F
import torch.nn as nn


def sample_rollout(model, prompt_ids, max_new_tokens):
    # 左 padding：不等长 prompt 靠右对齐，生成段一律从 L0 列开始，L0 之后的位置天然对齐。
    # 注意 PAD_ID == EOS_ID == 96，靠 id 分不出 padding，只能靠 attn_mask 标出真实 prompt 位。
    B = len(prompt_ids)
    L0 = max(len(p) for p in prompt_ids)
    device = next(model.parameters()).device
    out = torch.full((B, L0), PAD_ID, dtype=torch.long, device=device)
    attn_mask = torch.zeros((B, L0), dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids):
        out[i, L0 - len(p):] = torch.tensor(p, device=device)
        attn_mask [i, L0-len(p):] = 1
    # generate 是 @torch.no_grad 的，且会自己往 attn_mask 尾部补 1（新 token 全有效）
    return model.generate(out, L0+max_new_tokens,  EOS_ID,attn_mask=attn_mask)

def compute_logprobs(model, ids, attn_mask=None):
    # ids (B, S) - > (B, S)
    # [:, t] = log pi (ids[:, t] | ids[:,:t]); [:, 0] 无前驱置 0
    logits = model(ids, attn_mask=attn_mask)
    # 错位一位：第 t 个 token 的概率由第 t-1 位的 logits 预测（next-token 预测）
    pred = logits[:,:-1]
    tgt = ids[:,1:]
    V = logits.shape[-1]
    B, S = ids.shape
    nll = F.cross_entropy(pred.reshape(-1,V ), tgt.reshape(-1),
                           reduction="none",
                           )
    # [:,0] 保持 0（无前驱），与 k3 里 mask[:,0]=0 的契约对齐，首位不进 KL
    out = torch.zeros((B,S), dtype=nll.dtype, device=ids.device)
    out[:,1:] = -nll.reshape(B, S-1)
    return out

def compute_kl_k3(logp_theta, logp_ref, mask, delta_clamp=50.0):
    # 逐 token 的 k3 估计量，三个输入与输出同为 (B, S)。
    # k3 = e^Δ - Δ - 1，其中 Δ = logp_ref - logp_theta = log(π_ref/π_θ)。
    # 无偏：E_{π_θ}[e^Δ] = E[π_ref/π_θ] = 1，而 E[Δ] = -KL，故 E[k3] = KL。
    # 非负：e^x >= 1 + x 对任意实数成立。
    delta = logp_ref - logp_theta
    # float32 的 exp 在 Δ ≈ 88.7 处溢出；clamp 到 50 留出余量（e^50 ≈ 5.2e21，离 3.4e38 还远）
    delta = torch.clamp(delta, max=delta_clamp)
    k_3 = torch.exp(delta) - delta - 1
    # padding / prompt 位 mask 置 0
    return k_3 * mask
    
class Critic(nn.Module):
    # 与 RM 共用 LLaMA backbone，但 head 的语义不同，不能照抄 RM：
    #   RM head    = 排序头，bias=False（成对差把偏置消掉），只取末位 -> (B,)
    #   Critic head = 价值头，bias=True（要输出绝对标度，偏置不能消），每位都出值 -> (B, S)
    # (B, S) 是因为 P4 要把它和逐 token reward 相加算 advantage。
    def __init__(self, ckpt_path: str):
        super().__init__()
        ckpt = torch.load(ckpt_path)
        self.config = ckpt["config"]
        # 给 ckpt 盖溯源戳（对应 RM 的 "last"）。LLaMA 本身不读这个键，只在保存 critic ckpt 时留痕。
        self.config["pooling"] = "none"

        D = self.config["d_model"]
        self.scalar_head = nn.Linear(D, 1, bias=True)
        model = LLaMA(ckpt["vocab_size"], self.config["d_model"],self.config["n_layers"],self.config["n_heads"],self.config["n_kv_heads"],self.config["head_dim"],)
        self.model = model
        sd = ckpt["state_dict"]

        # 兼容两种 ckpt：带 scalar_head 的（critic 自存，整体加载）vs 裸 backbone（SFT/RM 存的，只填 backbone）
        if any(k.startswith("scalar_head.") for k in sd):
            self.load_state_dict(sd)
        else:
            self.model.load_state_dict(sd)

    def forward(self, input_ids, attn_mask=None):
        # return (B, S)
        h = self.model.get_backbone(input_ids, attn_mask)
        h = self.scalar_head(h)
        h = h.squeeze(-1)
        return h

def compute_rewards(rm, prompt_ids, response_ids, mask):
    # 把 RM 的序列级标量 r_φ 摊成 (B,S)：只在 response 末位给分，其余位（prompt 左 pad + gen 后尾部 pad）严格 0。
    # terminal-only 的理由：γ=1 时 G_t = Σ_{t'≥t} r_{t'} 要与长度解耦，且 G_0 精确对上目标 E[r_φ(x,y)]；
    # 若每位都给 r_φ，G_t 会带 (T−t+1) 因子 ⇒ 长度偏置，方向随 r_φ 符号（>0 学会拖长，<0 学会早停）。
    r_phi = rm.score_pair(prompt_ids, response_ids)
    B, S = mask.shape
    reward = torch.zeros((B, S), device = r_phi.device, dtype=r_phi.dtype)
    # 末位用位置累积定位：PAD_ID==EOS_ID==96，不能靠比值；且 EOS 截断后各行末位不同，写死 S-1 必错。
    last = mask.cumsum(-1).argmax(-1)
    reward[torch.arange(B, device=reward.device), last] = r_phi
    return reward

def forward_pass(actor, ref, rm, prompt_ids, max_new_tokens, beta):
    # 前向链路：生成 → 两份 logprob → k3 → 稀疏 RM 分 → 组装 r_t = −β·k3 + 1_{t=T}·r_φ。
    # 返回的 7 个张量全 (B,S)，供 P4 完整 step 直接取用。
    ids = sample_rollout(actor, prompt_ids, max_new_tokens)
    L0 = max(len(p) for p in prompt_ids)
    B, S = ids.shape
    dev = next(actor.parameters()).device  # nn.Module 无 .device，从参数取
    # 两张 mask 语义分层：attn_mask 管"看不看"（喂 attention，prompt 左 pad 段 0）；
    # resp_mask 管"算不算 / 给不给分"（喂 k3 与 compute_rewards，仅 response 段 1）。
    # 两者在 prompt 段都是 0，但 gen 段之外就分开（attn 从 prompt 起点起 1，resp 从 L0 起 1），不能合并。
    attn_mask = torch.zeros((B, S), device=dev, dtype=torch.long)
    resp_mask = torch.zeros((B, S), device=dev, dtype = torch.long)
    for i, p in enumerate(prompt_ids):
        attn_mask[i, L0-len(p):] = 1
    resp_mask[:, L0:] = 1

    # 梯度只从 actor 这条路进图：logp_theta 带 grad，logp_ref 在 no_grad 里断图。
    logp_theta = compute_logprobs(actor, ids, attn_mask=attn_mask)
    with torch.no_grad():
        logp_ref = compute_logprobs(ref, ids, attn_mask=attn_mask)

    # kl 是 logp_theta 的函数 ⇒ 带 grad（P4 靠它把梯度从 reward 回传到 actor）
    kl = compute_kl_k3(logp_theta, logp_ref, resp_mask)

    # response 段按 mask 取；P3 不截尾（EOS 截断留 P4）。rm 冻住 ⇒ eval + no_grad，r_φ 是常数。
    response_ids = []
    for i in range(B):
        response_ids.append(ids[i][resp_mask[i].bool()])
    rm.eval()
    with torch.no_grad():
        r_sparse = compute_rewards(rm, prompt_ids, response_ids, resp_mask)

    reward = -beta * kl + r_sparse  # 梯度只从 −β·kl 回传，r_φ 不贡献（详见 check_forward_pass 判据 B）
    return dict(ids=ids, attn_mask=attn_mask, resp_mask=resp_mask,
                logp_theta=logp_theta, logp_ref=logp_ref, kl=kl, reward=reward)

def advantage(reward, value):
    # γ=1 λ=1 的 GAE 退化式：Â_t = G_t − V_t，G_t = Σ_{t'≥t} r_{t'}（后缀和）。
    # reward 只末位非零 ⇒ TD 残差只有末位 ≠0，且权重 (γλ)^l ≡ 1 ⇒ 递推坍成纯后缀和，V 只作 baseline 减这一次。
    # 后缀和 = 反向 cumsum，再 flip 回原序（flip 少做一次会得到对称错位的结果）。
    # 不收 resp_mask：mask 外保持 G_t−V_t 原值，由下游两条 loss 各自乘 mask 抹掉。
    G_t_reverse = torch.cumsum(torch.flip(reward, dims=[-1]), dim=-1)
    G_t = torch.flip(G_t_reverse, dims=[-1])
    return G_t - value

def normalization(A_t, epsilon, resp_mask):
    # Â 归一化：把 advantage 压到 ~N(0,1)。不归一化时各 batch 的 Â 尺度差很大，
    # 同一个 lr 下梯度步长会跟着飘（BT 只约束 r_φ 的差 ⇒ 绝对标度自由，必须先定住）。
    # μ/σ 只在 resp_mask==1（response 段）上统计，且 .detach()：统计量本身不能带梯度，
    # 否则「除 σ」会额外引一条对全部 token 的梯度通路。
    # 整式除 (A_t−μ)/(σ+ε)，不是 A_t − μ/(σ+ε)；mask 外也被变换，下游乘 mask 抹掉。
    sel = A_t[resp_mask.bool()].detach()
    mu = sel.mean()
    sigma = sel.std()
    return (A_t - mu) / (sigma + epsilon)

def actor_loss(logp_theta, logp_old, adv, resp_mask, eps):
    # clipped surrogate：L^CLIP = mean[min(ρÂ, clip(ρ, 1±ε)Â)]，ρ = π_θ/π_old = exp(logp_θ − logp_old)。
    # min 取悲观分支 ⇒ ρ 偏离 1 超过 ε 后该 token 的梯度被掐断，这就是 trust region 的落地形式。
    # 返回 −L^CLIP：调用方直接 .backward() 即最大化 L^CLIP，train 侧不再取负。
    # 分子也必须乘 mask —— adv 走「mask 外保持原值」分支、prompt/pad 位非 0，
    # 只 mask 分母会把那些位的 ρ·Â 混进均值。
    ro = torch.exp(logp_theta - logp_old)
    loss = torch.min(ro*adv, torch.clip(ro, 1-eps, 1+eps)*adv)[resp_mask.bool()].mean()
    return -loss

def critic_loss(value, value_old, adv, resp_mask):
    # 价值回归：target = Â + V_old = G_t（advantage 定义的反解），masked MSE。
    # V_old 是调用方传进来的快照（常量）⇒ 梯度只经 value 一项，critic 不会去追自己的移动目标。
    # 分母用 resp_mask.sum() 而非元素数：prompt/pad 位被乘 0，也必须从分母里去掉。
    target = adv + value_old
    loss = (resp_mask * (value - target) ** 2).sum() / resp_mask.sum()
    return loss

def ppo_step(out, critic, actor, opt_critic, opt_actor,eps_norm=1e-8, K=4,eps_clip=0.2, c_v=0.5):
    # 入参 out = forward_pass 的 7 键 dict。串起「快照 → GAE → 归一化 → K 轮更新」。
    # 两个快照在 K 循环外各算一次：logp_old（ratio 分母）与 V_old（advantage 的 baseline + 回归 target 的一部分）。
    # out["reward"] 必须 .detach() —— 否则 Â 带图，K>1 时会撞 backward a second time。
    logp_old = out["logp_theta"].detach()
    V_old = critic(out["ids"], out["attn_mask"]).detach()
    adv_raw = advantage(out["reward"].detach(), V_old)
    adv = normalization(adv_raw, eps_norm, out["resp_mask"])
    # K 轮复用同一批 rollout：随 k 递增这批数据越 off-policy，靠 clip 兜住（k 太大则 clip 饱和、白算）。
    # 两条 loss 因快照全为常量而不共享参数 ⇒ 合并 backward ≡ 分别 backward。
    # 两个 opt 各自 zero_grad；步进顺序（critic 先）不影响正确性，固定下来便于复现。
    for _ in range(K):
        logp_new = compute_logprobs(actor, out["ids"], out["attn_mask"])
        V_phi = critic(out["ids"], out["attn_mask"])
        a_loss = actor_loss(logp_new, logp_old, adv, out["resp_mask"],eps_clip)
        c_loss = critic_loss(V_phi, V_old, adv, out["resp_mask"])
        L = a_loss + c_v * c_loss
        opt_actor.zero_grad()
        opt_critic.zero_grad()
        L.backward()
        opt_critic.step()
        opt_actor.step()

    

