"""check_ppo_step.py — ppo_step 集成层的静默失败点检查（只读；不碰 ckpt / 真实模型）

不重验子函数（actor_loss / critic_loss / advantage / normalization 各有自己的 check）。
这里只验"集成"才引入的东西：快照时机、detach 边界、optimizer 接线、zero_grad。
用 Dummy actor/critic，CPU 上秒跑。
"""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path += [str(Path(__file__).resolve().parent)]
import torch
import torch.nn as nn
from ppo_lm import (ppo_step, advantage, normalization, actor_loss,
                    critic_loss, compute_logprobs)

FAIL = []
V_, D_, B_, S_ = 8, 4, 2, 6


def report(tag, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:<32} {detail}")
    if not ok:
        FAIL.append(tag)


class DummyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V_, D_)
        self.head = nn.Linear(D_, V_)

    def forward(self, ids, attn_mask=None):
        return self.head(self.emb(ids))                       # (B,S,V) logits


class DummyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V_, D_)
        self.head = nn.Linear(D_, 1)

    def forward(self, ids, attn_mask=None):
        return self.head(self.emb(ids)).squeeze(-1)           # (B,S)


class Spy(nn.Module):
    """数前向调用次数"""
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.n = 0

    def forward(self, ids, attn_mask=None):
        self.n += 1
        return self.m(ids, attn_mask)


def make_out():
    """模拟 forward_pass 的 7 键出口；reward 真的依赖 logp_theta（与真实链路同构）"""
    ids = torch.randint(0, V_, (B_, S_))
    attn_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    resp_mask = torch.tensor([[0.0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    lt = torch.randn(B_, S_, requires_grad=True)              # forward_pass 的 logp_theta 出口
    reward = -0.3 * torch.exp(lt)                             # 依赖 logp_theta ⇒ 带图
    return dict(ids=ids, attn_mask=attn_mask, resp_mask=resp_mask,
                logp_theta=lt, reward=reward), lt


def build(seed=0):
    torch.manual_seed(seed)
    actor, critic = DummyActor(), DummyCritic()
    oa = torch.optim.SGD(actor.parameters(), lr=0.05)
    oc = torch.optim.SGD(critic.parameters(), lr=0.05)
    return actor, critic, oa, oc


def snap(m):
    return [p.detach().clone() for p in m.parameters()]


def same(x, y):
    return all((a - b).abs().max().item() == 0 for a, b in zip(x, y))


def changed(x, y):
    return any((a - b).abs().max().item() > 0 for a, b in zip(x, y))


# ---------- 1 detach 边界：forward_pass 出口不应收到梯度 ----------
out, lt = make_out()
actor, critic, oa, oc = build()
ppo_step(out, critic, actor, oc, oa, K=2)
g1 = "None" if lt.grad is None else f"{float(lt.grad.abs().sum()):.3e}"
report("1 detach 边界(出口无梯度)", lt.grad is None, f"logp_theta.grad = {g1}")


def step_bad_reward(out, critic, actor, oc, oa, K=2):
    """反证：reward 不 detach"""
    logp_old = out["logp_theta"].detach()
    V_old = critic(out["ids"], out["attn_mask"]).detach()
    adv = normalization(advantage(out["reward"], V_old), 1e-8, out["resp_mask"])   # 漏 detach
    for _ in range(K):
        logp_new = compute_logprobs(actor, out["ids"], out["attn_mask"])
        V_phi = critic(out["ids"], out["attn_mask"])
        L = actor_loss(logp_new, logp_old, adv, out["resp_mask"], 0.2) \
            + 0.5 * critic_loss(V_phi, V_old, adv, out["resp_mask"])
        oa.zero_grad(); oc.zero_grad(); L.backward(); oc.step(); oa.step()


out_b, lt_b = make_out()
a_b, c_b, oa_b, oc_b = build()
# K=1：反证故意不 detach，reward 的图 K 轮复用会在第 2 轮炸「backward a second time」，
# 那是同一病根的另一种表现，但会挡住正面判据。这里只问"梯度能否摸到出口"，K=1 即可。
step_bad_reward(out_b, c_b, a_b, oc_b, oa_b, K=1)
report("1' 反证·不 detach reward", lt_b.grad is not None,
       f"grad = {float(lt_b.grad.abs().sum()):.3e}  (应非 None)")

# ---------- 2 快照在循环外：critic 应被调 1+K 次 ----------
K = 3
out, _ = make_out()
actor, critic, oa, oc = build()
spy = Spy(critic)
ppo_step(out, spy, actor, oc, oa, K=K)
report("2 V_old 在循环外(1+K 次)", spy.n == 1 + K, f"critic 调用 {spy.n} 次，应 {1 + K}")


def step_bad_vold(out, critic, actor, oc, oa, K=3):
    """反证：V_old 算在循环内"""
    logp_old = out["logp_theta"].detach()
    for _ in range(K):
        V_old = critic(out["ids"], out["attn_mask"]).detach()          # 错位：循环内
        adv = normalization(advantage(out["reward"].detach(), V_old), 1e-8, out["resp_mask"])
        logp_new = compute_logprobs(actor, out["ids"], out["attn_mask"])
        V_phi = critic(out["ids"], out["attn_mask"])
        L = actor_loss(logp_new, logp_old, adv, out["resp_mask"], 0.2) \
            + 0.5 * critic_loss(V_phi, V_old, adv, out["resp_mask"])
        oa.zero_grad(); oc.zero_grad(); L.backward(); oc.step(); oa.step()


out, _ = make_out()
actor, critic, oa, oc = build()
spy2 = Spy(critic)
step_bad_vold(out, spy2, actor, oc, oa, K=K)
report("2' 反证·V_old 在循环内(2K 次)", spy2.n == 2 * K, f"critic 调用 {spy2.n} 次，应 {2 * K}")

# ---------- 3 optimizer 参数组不交叉 ----------
actor, critic, oa, oc = build()
a_ids = {id(p) for p in actor.parameters()}
c_ids = {id(p) for p in critic.parameters()}
oa_ids = {id(p) for g in oa.param_groups for p in g["params"]}
oc_ids = {id(p) for g in oc.param_groups for p in g["params"]}
report("3 opt_actor 只含 actor 参数", oa_ids == a_ids, f"{len(oa_ids)} vs {len(a_ids)}")
report("3 opt_critic 只含 critic 参数", oc_ids == c_ids, f"{len(oc_ids)} vs {len(c_ids)}")
report("3 两组零交集", not (oa_ids & oc_ids), f"交集 = {len(oa_ids & oc_ids)}")
oa_bad = torch.optim.SGD(list(actor.parameters()) + list(critic.parameters()), lr=0.05)
oa_bad_ids = {id(p) for g in oa_bad.param_groups for p in g["params"]}
report("3' 反证·混合 opt", oa_bad_ids != a_ids, f"{len(oa_bad_ids)} vs {len(a_ids)}")

# ---------- 4 zero_grad：脏梯度不影响结果 ----------
out, _ = make_out()
a1, c1, oa1, oc1 = build()
ppo_step(out, c1, a1, oc1, oa1, K=2)
a2, c2, oa2, oc2 = build()
for p in list(a2.parameters()) + list(c2.parameters()):
    p.grad = torch.full_like(p, 1e3)
ppo_step(out, c2, a2, oc2, oa2, K=2)
report("4 zero_grad 清脏梯度", same(snap(a1), snap(a2)), "脏梯度未生效" if same(snap(a1), snap(a2)) else "参数被污染")


def step_bad_nograd(out, critic, actor, oc, oa, K=2):
    """反证：去掉 zero_grad"""
    logp_old = out["logp_theta"].detach()
    V_old = critic(out["ids"], out["attn_mask"]).detach()
    adv = normalization(advantage(out["reward"].detach(), V_old), 1e-8, out["resp_mask"])
    for _ in range(K):
        logp_new = compute_logprobs(actor, out["ids"], out["attn_mask"])
        V_phi = critic(out["ids"], out["attn_mask"])
        L = actor_loss(logp_new, logp_old, adv, out["resp_mask"], 0.2) \
            + 0.5 * critic_loss(V_phi, V_old, adv, out["resp_mask"])
        L.backward()                      # 不 zero_grad
        oc.step(); oa.step()


out, _ = make_out()
d1, e1, oad1, ocd1 = build()
step_bad_nograd(out, e1, d1, ocd1, oad1, K=2)
d2, e2, oad2, ocd2 = build()
for p in list(d2.parameters()) + list(e2.parameters()):
    p.grad = torch.full_like(p, 1e3)
step_bad_nograd(out, e2, d2, ocd2, oad2, K=2)
report("4' 反证·去掉 zero_grad", changed(snap(d1), snap(d2)), "脏梯度确实生效了")

# ---------- 5 c_v 接线 + 两 loss 不相交 ----------
out, _ = make_out()
actor, critic, oa, oc = build()
ab, cb = snap(actor), snap(critic)
ppo_step(out, critic, actor, oc, oa, K=2, c_v=0.0)
report("5 c_v=0 → critic 冻结", same(cb, snap(critic)), "critic 未被 actor 那条路污染")
report("5 c_v=0 → actor 仍更新", changed(ab, snap(actor)), "actor 学到了")

# ---------- 6 边界 K=0 ----------
out, _ = make_out()
actor, critic, oa, oc = build()
ab = snap(actor)
ppo_step(out, critic, actor, oc, oa, K=0)
report("6 K=0 → 参数不变", same(ab, snap(actor)), "无更新")

# ---------- 7 K 次更新确实发生 ----------
out, _ = make_out()
actor, critic, oa, oc = build()
ab = snap(actor)
ppo_step(out, critic, actor, oc, oa, K=2)
report("7 K=2 → 参数被更新", changed(ab, snap(actor)), "actor 变了")

print()
print(f"== {len(FAIL)} FAIL: {FAIL} ==" if FAIL else "== ALL PASS ==")
