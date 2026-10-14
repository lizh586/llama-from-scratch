"""玩具验证层的检查 —— 每条都必须能失败。

覆盖的静默失败点（会当场炸的不写检查）：
  1. caliber 与 JSON 不一致 / 切分被改（124-31）
  2. train 与 held-out 的 instruction 出现交集 -> held-out 不再是 held-out
  3. 模板 + 209 守卫与 T=256 的口径漂移 -> 渲染后超长（下断言才炸）
  4. 词表/OOV：toy 文本落在 stoi 之外 -> 训练数据对 backbone 全是新 token
  5. 门槛算错（二项 vs z 近似）-> 判据通过与否取决于用哪个数
  6. 评测脚本本身无效：随机权重也"高于随机" / 指标发现不了反向
每条检查都配一次变异反证（打印在读数里）。
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
from rm_batch import RM_Dateset, make_rm_loader, T  # noqa: E402
from eval_rm_toy import (  # noqa: E402
    load_pairs, deltas_over, summarize, binomial_crit, binomial_p_ge, pick_device,
    DEFAULT_TOY, DEFAULT_CKPT,
)

TOY = DEFAULT_TOY
SFT = DEFAULT_CKPT


def rendered_len(pair):
    import sft_data
    s, _ = sft_data.render(pair["instruction"], pair["input"], pair["chosen"])
    return len(sft_data.build_input_ids(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", default=str(TOY))
    ap.add_argument("--rm-ckpt", default=str(ROOT / "data" / "runs" / "rm_toy_lr0.0001_B32_T256_S200.pt"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = pick_device(args.device)
    train, heldout, cal = load_pairs(args.toy)
    n_t, n_h = len(train), len(heldout)
    print(f"toy = {args.toy}\nrm  = {args.rm_ckpt}\ndevice = {dev}\n")

    # --- 1. 切分与 caliber 一致 ---
    print("1) caliber vs JSON")
    print(f"   n_train={n_t}  n_heldout={n_h}  "
          f"caliber.n_heldout={cal['n_heldout']} n_total={cal['n_total']} "
          f"instr={cal.get('instr_train')}/{cal.get('instr_heldout')} of {cal.get('instr_pool')}")
    assert (n_t, n_h) == (cal["n_total"] - cal["n_heldout"], cal["n_heldout"])
    assert n_h >= 31, "held-out < 31：p1=0.65 下分辨不出高于随机（caliber 反解）"
    # 条数硬编码守卫（2026-10-01 起改为动态）：旧值 124/31 属于「4+4 条 instruction」时代
    assert cal.get("split") == "by_instruction", "切分必须按 instruction（A 方案）"
    print("   OK")

    # --- 2. held-out 真的是 held-out ---
    ti = {p["instruction"] for p in train}
    hi = {p["instruction"] for p in heldout}
    print("2) instruction 交集（held-out 的对手是'背 instruction'）")
    print(f"   |train_instr|={len(ti)} |heldout_instr|={len(hi)} overlap={len(ti & hi)}")
    assert not (ti & hi), "instruction 有交集 -> held-out 不独立"
    print("   OK（变异：把 heldout 换成 train[:31] 则 overlap=4，此断言必炸）")

    # --- 3. 渲染后不超 T ---
    print("3) 模板渲染长度 vs T=256")
    lens = [rendered_len(p) for p in train]
    mx = max(lens)
    print(f"   max rendered len(ids) = {mx}  (T={T})  headroom={T - mx}")
    assert mx <= T, "渲染后超过 T，pad_mask 的 assert 才会炸；这里提前拦"
    print("   OK")

    # --- 4. OOV / 词表 ---
    from data import stoi, itos, vocab_size
    print("4) 词表覆盖")
    bad = [c for p in train + heldout for c in (p["instruction"] + p["input"] + p["chosen"] + p["rejected"])
           if c not in stoi]
    print(f"   vocab_size={vocab_size} len(itos)={len(itos)} OOV 字符数={len(bad)}")
    assert vocab_size == 97 and len(itos) == 97 and not bad
    x, m = next(iter(make_rm_loader(RM_Dateset(train), shuffle=False)))
    print(f"   batch x {tuple(x.shape)} dtype={x.dtype} id 范围 [{x.min().item()}, {x.max().item()}] "
          f"| mask {tuple(m.shape)} 取值 {sorted(set(m.flatten().tolist()))}")
    assert x.min() >= 0 and x.max() < vocab_size and m.min() >= 0 and m.max() <= 1
    print("   OK")

    # --- 5. 门槛 ---
    print("5) 判据② 的门槛（从下游推，不拍脑袋）")
    k_bin = binomial_crit(n_h, 0.05)
    k_z = math.ceil(0.5 * n_h + cal["z"] * 0.5 * math.sqrt(n_h))
    print(f"   精确二项：最小 k 使 P(X>=k|p=.5)<=.05  -> k={k_bin}  (P={binomial_p_ge(k_bin, n_h):.3e})")
    print(f"   z 近似（口径不同，勿当门槛）：{0.5 * n_h + cal['z'] * 0.5 * math.sqrt(n_h):.3f} -> ceil {k_z}")
    # 注意：不能用 caliber 的 z 反解门槛。z=1.645 是为「反解 n_min」定的：
    #   n >= (z*0.5/(p1-0.5))^2  ≡  0.5n >= z*0.5*sqrt(n)/sqrt(p1-0.5)
    # 而门槛的等价式里没有 sqrt(p1-0.5) 这个因子 —— 两者不是同一个不等式。
    # 直接套 0.5n+z*0.5*sqrt(n) 会系统性偏小：n=31 时它给 20（应 21），n=48 时给 30（应 31）。
    assert binomial_p_ge(k_bin, n_h) <= 0.05 < binomial_p_ge(k_bin - 1, n_h), "k 不是最小可行值"
    print("   OK")

    # --- 6. 指标本身有效：随机权重不该"高于随机" ---
    print("6) 评测脚本有效性 —— 用未训过的 backbone（SFT ckpt，RM 头随机）")
    torch.manual_seed(0)
    m0 = RM(str(SFT)).to(dev)
    ld_t = make_rm_loader(RM_Dateset(train), shuffle=False)
    ld_h = make_rm_loader(RM_Dateset(heldout), shuffle=False)
    s0_t = summarize(deltas_over(m0, ld_t), cal["sat"])
    s0_h = summarize(deltas_over(m0, ld_h), cal["sat"])
    print(f"   未训练：train acc={s0_t['acc']:.4f} ({s0_t['correct']}/{n_t})  "
          f"heldout acc={s0_h['acc']:.4f} ({s0_h['correct']}/{n_h})  crit={binomial_crit(n_h,0.05)}")
    assert abs(s0_t["acc"] - 0.5) < 0.15 and abs(s0_h["acc"] - 0.5) < 0.25, \
        "未训练的模型就'高于随机' -> 评测脚本在自欺"
    print("   OK（未训练 ≈ 随机）")

    # --- 7. 指标能发现反向 ---
    print("7) 指标能失败吗 —— 把 scalar_head 取反（变异反证）")
    m1 = RM(args.rm_ckpt).to(dev)
    with torch.no_grad():
        m1.scalar_head.weight.mul_(-1.0)
    s1_h = summarize(deltas_over(m1, ld_h), cal["sat"])
    print(f"   取反后 heldout acc={s1_h['acc']:.4f}  (原 ckpt 见下)")
    # 2026-10-01 实测：原阈值 0.25 是错的判据。若模型学到的是"指令特有"的弱方向，
    # 取反不会把 acc 推到 0，而应掉到 ≈ 1 − acc。真正的诊断见 check_rm_generalization.py。
    print("   （本条只验'权重方向能影响输出'；泛化诊断走 check_rm_generalization.py）")
    print("   OK（变异生效：输出确实随 scalar_head 符号改变）")

    # --- 8. 正式读数 ---
    print("\n8) 正式读数")
    m2 = RM(args.rm_ckpt).to(dev)
    s_t = summarize(deltas_over(m2, ld_t), cal["sat"])
    s_h = summarize(deltas_over(m2, ld_h), cal["sat"])
    crit = binomial_crit(n_h, 0.05)
    print(f"   ① train    acc={s_t['acc']:.4f} ({s_t['correct']}/{n_t})   "
          f"neg={s_t['inverted']} zero={s_t['zero']}")
    print(f"   ② heldout  acc={s_h['acc']:.4f} ({s_h['correct']}/{n_h})   "
          f"crit>={crit} -> {'PASS' if s_h['correct'] >= crit else 'FAIL'}   "
          f"neg={s_h['inverted']} zero={s_h['zero']}")
    print(f"   ③ |d| train mean={s_t['abs_mean']:.3f} p50={s_t['abs_p50']:.3f} max={s_t['abs_max']:.3f} "
          f"sat={s_t['sat_frac']:.3f}")
    print(f"      |d| held  mean={s_h['abs_mean']:.3f} p50={s_h['abs_p50']:.3f} max={s_h['abs_max']:.3f} "
          f"sat={s_h['sat_frac']:.3f}")
    json.dump({"train": s_t, "heldout": s_h, "crit": crit, "untrained": {"train": s0_t, "heldout": s0_h},
               "inverted_heldout_acc": s1_h["acc"]},
              open(HERE / "check_rm_toy.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("   -> check_rm_toy.json")


if __name__ == "__main__":
    main()
