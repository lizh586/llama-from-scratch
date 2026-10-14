"""玩具偏好集上的 RM 验证层。

三条读数：
  delta      (n,) = r_chosen - r_rejected，逐 pair
  acc        delta > 0 的比例（pairwise accuracy）
  sat_frac   |delta| > SAT 的比例（饱和分层；SAT 由 toy_pairs.caliber 下发）

判据②「held-out 高于随机」的门槛不拍脑袋：由 caliber（单侧 alpha=0.05, n=31）
反解「至少对多少条才能在 alpha 下拒绝 p=0.5」，见 binomial_crit。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from paths import ROOT  # noqa: E402
sys.path += [str(HERE), str(HERE.parent / "pipeline"), str(HERE.parent / "sft")]

from rm import RM  # noqa: E402
from rm_batch import RM_Dateset, make_rm_loader  # noqa: E402

DEFAULT_TOY = ROOT / "data" / "rlhf" / "toy_pairs.json"
DEFAULT_CKPT = ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt"
DEFAULT_SAT = 5.0


def pick_device(name=None):
    """CUDA 可用就用 GPU（项目约定：训练默认走 GPU）。verify 脚本与 train 共用。"""
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pairs(path=DEFAULT_TOY):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cal = dict(data.get("caliber") or {})
    cal.setdefault("sat", DEFAULT_SAT)
    cal["_path"] = str(path)
    return data["train"], data["heldout"], cal


def binomial_p_ge(k, n, p=0.5):
    """P(X >= k)，X ~ Bin(n, p)。不依赖 scipy：手写尾概率。"""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def binomial_crit(n, alpha=0.05):
    """最小的 k，使 P(X >= k | p=0.5) <= alpha —— 即判"高于随机"的正确率门槛。"""
    for k in range(n + 1):
        if binomial_p_ge(k, n) <= alpha:
            return k
    return None


@torch.no_grad()
def deltas_over(model, loader, device=None):
    """返回每个 pair 的 delta（第 0 条 = chosen，第 1 条 = rejected）。

    device=None -> 取 model 参数所在设备（x/m 是 CPU 上造的，必须显式搬）。
    """
    dev = torch.device(device) if device else next(model.parameters()).device
    out = []
    model.eval()
    for x, m in loader:
        x, m = x.to(dev), m.to(dev)
        delta = model(x[:, 0], m[:, 0]) - model(x[:, 1], m[:, 1])
        out += delta.tolist()
    return out


def summarize(deltas, sat=DEFAULT_SAT):
    n = len(deltas)
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = n - pos - neg
    a = sorted(abs(d) for d in deltas)
    return {
        "n": n,
        "correct": pos,
        "acc": pos / n if n else None,
        "inverted": neg,
        "zero": zero,
        "delta_mean": sum(deltas) / n if n else None,
        "abs_mean": sum(a) / n if n else None,
        "abs_p50": a[len(a) // 2] if a else None,
        "abs_p90": a[min(len(a) - 1, int(0.9 * len(a)))] if a else None,
        "abs_max": a[-1] if a else None,
        "sat_frac": sum(1 for d in a if d > sat) / n if n else None,
    }


def print_row(tag, s, crit, sat):
    ok = "PASS" if s["correct"] >= crit else "FAIL"
    print(
        f"{tag:18s} n={s['n']:3d}  correct={s['correct']:3d}/{s['n']:<3d}  "
        f"acc={s['acc']:.4f}  (crit>={crit} -> {ok})  "
        f"neg={s['inverted']:2d} zero={s['zero']:2d}"
    )
    print(
        f"{tag:18s} |d| mean={s['abs_mean']:.4f} p50={s['abs_p50']:.4f} "
        f"p90={s['abs_p90']:.4f} max={s['abs_max']:.4f}  "
        f"sat(|d|>{sat:g})={s['sat_frac']:.3f}  d_mean={s['delta_mean']:+.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="RM 的 backbone 初始化 ckpt")
    ap.add_argument("--toy", default=str(DEFAULT_TOY))
    ap.add_argument("--rm-ckpt", default=None, help="已训好的 RM ckpt（带 scalar_head.* 前缀键）")
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    train_pairs, heldout_pairs, cal = load_pairs(args.toy)
    n_h = len(heldout_pairs)
    assert cal.get("n_heldout") == n_h, f"caliber 写 {cal.get('n_heldout')}，实际 {n_h}"

    crit_h = binomial_crit(n_h, alpha=0.05)
    crit_t = binomial_crit(len(train_pairs), alpha=0.05)
    print(
        f"caliber: p1={cal.get('p1')} alpha=0.05  n_heldout={n_h}  "
        f"crit_heldout P(X>={crit_h})={binomial_p_ge(crit_h, n_h):.3e}  "
        f"(若写 z 门槛 {0.5 * n_h + cal['z'] * 0.5 * math.sqrt(n_h):.2f})"
    )

    kwargs = {"shuffle": False, "drop_last": False}
    if args.batch:
        kwargs["batch"] = args.batch
    ds_t = RM_Dateset(train_pairs)
    ds_h = RM_Dateset(heldout_pairs)
    ld_t = make_rm_loader(ds_t, **kwargs)
    ld_h = make_rm_loader(ds_h, **kwargs)

    model = RM(args.rm_ckpt or args.ckpt)

    d_t = deltas_over(model, ld_t)
    d_h = deltas_over(model, ld_h)
    s_t = summarize(d_t, cal["sat"])
    s_h = summarize(d_h, cal["sat"])

    print(f"\nsource = {args.rm_ckpt or args.ckpt}")
    print("--- 判据① 训练 pair 排序（验实现；不验泛化，必然通过） ---")
    print_row("train", s_t, crit_t, cal["sat"])
    print("--- 判据② held-out 排序（验不是纯记忆） ---")
    print_row("heldout", s_h, crit_h, cal["sat"])
    print("--- 判据③ Δ 分布（看饱和，配合早停） ---")
    print(f"sat_frac train={s_t['sat_frac']:.3f}  heldout={s_h['sat_frac']:.3f}")

    return {
        "caliber": cal,
        "crit_heldout": crit_h,
        "crit_train": crit_t,
        "train": s_t,
        "heldout": s_h,
    }


if __name__ == "__main__":
    main()
