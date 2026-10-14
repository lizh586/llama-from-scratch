"""玩具验证的泛化诊断。

存在理由（2026-10-01 实测）：`check_rm_toy.py` 第 7 条的"head 取反 → acc 应掉到 ≈0"
在真实读数上失败了。查下去发现不是脚本 bug，而是**held-out 确实没学到规则**：

  train(124)  acc = 1.0000  (124/124)  |delta| 全部 > 5（饱和）
  held-out(31) acc = 0.3871 (12/31)     ← 低于随机
  未训练       held-out acc = 0.5161 (16/31)
  head 取反    held-out acc = 0.6129 (19/31)

读数：**训练把 held-out 从 0.516 拉到 0.387（负迁移）**，train 满分 + held-out 低于随机 =
典型的"只记住没学会"签名，本次是**实测命中**。

本脚本把该诊断固定下来：deltas 的可分解结构 + 三条对照读数。
判据：**泛化要求** held-out acc ≥ 门槛（31 条时 = 21，由 caliber 反解），且 head 取反后
held-out acc 应掉到 ≈ 1 − 训练后（说明决定正负的确实是被训练的那个方向）。
"""
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
from paths import ROOT  # noqa: E402
sys.path += [str(HERE), str(HERE.parent / "pipeline"), str(HERE.parent / "sft")]
from rm import RM  # noqa: E402
from rm_batch import RM_Dateset, make_rm_loader  # noqa: E402
from eval_rm_toy import load_pairs, binomial_crit  # noqa: E402

SFT = ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt"
TRAINED = ROOT / "data" / "runs" / "rm_lr1e-04_B32_T256_S200.pt"


def scores(model, loader):
    """返回 (n,2) 的分数矩阵：第 0 列 chosen，第 1 列 rejected。"""
    rows = []
    model.eval()
    with torch.no_grad():
        for x, m in loader:
            rows.append(torch.stack([model(x[:, 0], m[:, 0]), model(x[:, 1], m[:, 1])], dim=1))
    return torch.cat(rows)


def acc_of(S):
    pos = int(((S[:, 0] - S[:, 1]) > 0).sum())
    return pos / len(S), pos


def report(tag, S):
    c, r = S[:, 0], S[:, 1]
    d = c - r
    a, pos = acc_of(S)
    print(f"\n[{tag}]")
    print(f"  chosen   mean={c.mean():+.4f} std={c.std():.4f} min={c.min():+.4f} max={c.max():+.4f}")
    print(f"  rejected mean={r.mean():+.4f} std={r.std():.4f} min={r.min():+.4f} max={r.max():+.4f}")
    print(f"  delta    mean={d.mean():+.4f} std={d.std():.4f} min={d.min():+.4f} max={d.max():+.4f}")
    print(f"  acc={a:.4f}  ({pos}/{len(d)})   |delta|>5 比例={float((d.abs() > 5).float().mean()):.4f}")
    print(f"  分解: 组均值差={c.mean() - r.mean():+.4f}  配对内 std(d)={d.std():.4f}")


def main():
    train, heldout, cal = load_pairs()
    n_h = len(heldout)
    crit = binomial_crit(n_h)
    ld_t = make_rm_loader(RM_Dateset(train), shuffle=False)
    ld_h = make_rm_loader(RM_Dateset(heldout), shuffle=False)

    print(f"held-out 门槛（caliber 反解）= {crit}/{n_h}")

    m0 = RM(str(SFT))
    S0t, S0h = scores(m0, ld_t), scores(m0, ld_h)
    report("未训练 · train", S0t)
    report("未训练 · held-out", S0h)

    m1 = RM(str(TRAINED))
    S1t, S1h = scores(m1, ld_t), scores(m1, ld_h)
    report("训练后 · train", S1t)
    report("训练后 · held-out", S1h)

    with torch.no_grad():
        m1.scalar_head.weight.mul_(-1.0)
    S1n = scores(m1, ld_h)
    report("训练后·head 取反 · held-out", S1n)

    a0t, _ = acc_of(S0t); a0h, k0h = acc_of(S0h)
    a1t, _ = acc_of(S1t); a1h, k1h = acc_of(S1h)
    a1n, k1n = acc_of(S1n)

    print("\n=== 判读 ===")
    print(f"  train      {a0t:.4f} -> {a1t:.4f}   （训练集必须满分；这条不验泛化）")
    print(f"  held-out   {a0h:.4f} -> {a1h:.4f}   （{k0h}/{n_h} -> {k1h}/{n_h}，门槛 {crit}）")
    print(f"  head 取反  {a1n:.4f}   （若学到真信号应 ≈ {1 - a1h:.4f}）")
    print(f"  泛化判据：{'PASS' if k1h >= crit else 'FAIL'}  "
          f"（{'高于随机' if k1h >= crit else '未达门槛；低于/接近随机 = 只记住没学会'}）")
    print(f"  负迁移检查：held-out {a1h - a0h:+.4f}（训练后低于训练前 = 负迁移，比随机更坏）")


if __name__ == "__main__":
    main()
