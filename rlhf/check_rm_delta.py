"""compute_delta 的语义判据 —— 训练路径 vs 推理路径同口径，外加非退化与反证。

两条独立路径算同一个 delta：
  T) 训练路径：RM_Dateset -> loader -> compute_delta(model, x, m)
  I) 推理路径：render 切点 + RM.score_pair（9/27 已验与训练路径逐位相同）
方向写反（r_r - r_c）或 chosen/rejected 取反，T 与 I 就不同 —— 这是本脚本唯一的靶子。
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(HERE))

from train_rm import compute_delta  # noqa: E402
from rm import RM  # noqa: E402
from rm_batch import RM_Dateset, make_rm_loader  # noqa: E402  (顺带注入 pipeline/sft 的 sys.path)
from sft_data import render, build_input_ids  # noqa: E402

CKPT = str(ROOT / "data" / "runs" / "sft_lr1e-05_B32_T256_S1782.pt")

PAIRS = [
    {"instruction": "Say hello.", "input": "", "chosen": "Hello!", "rejected": "Goodbye."},
    {"instruction": "Say hello.", "input": "", "chosen": "Hi there, nice to meet you.", "rejected": "No."},
    {"instruction": "Reverse the string.", "input": "abc", "chosen": "cba", "rejected": "abc"},
    {"instruction": "Add the numbers.", "input": "1 2", "chosen": "3", "rejected": "12"},
]


def split_ids(pair, key):
    """render 的切点 n 是字符数；char-level 下 1 char = 1 token，故 ids[:n] 即 prompt 段。"""
    s, n = render(pair["instruction"], pair["input"], pair[key])
    ids = build_input_ids(s)
    return torch.tensor(ids[:n]), torch.tensor(ids[n:])


def main():
    torch.manual_seed(0)
    model = RM(CKPT)
    ok = True

    with torch.no_grad():
        # --- 路径 T ---
        ds = RM_Dateset(PAIRS)
        loader = make_rm_loader(ds, shuffle=False)  # 顺序必须与 PAIRS 对齐，否则逐位比较无意义
        x, m = next(iter(loader))
        delta_t = compute_delta(model, x, m)

        # --- 路径 I ---
        pc, rc = [], []
        pr, rr = [], []
        for p in PAIRS:
            a, b = split_ids(p, "chosen")
            pc.append(a)
            rc.append(b)
            a, b = split_ids(p, "rejected")
            pr.append(a)
            rr.append(b)
        delta_i = model.score_pair(pc, rc) - model.score_pair(pr, rr)

    print("delta_t =", delta_t.tolist())
    print("delta_i =", delta_i.tolist())

    # --- 判据 1：两条路径同口径 ---
    max_diff = float((delta_t - delta_i).abs().max())
    exact = torch.equal(delta_t, delta_i)
    same = torch.allclose(delta_t, delta_i, atol=1e-6)
    ok &= same
    print(f"[1] path T vs path I   max|diff| = {max_diff:.3e}  exact={exact}  {'OK' if same else 'FAIL'}")

    # --- 判据 2：非退化 —— delta 若全近似 0，方向错误就检不出来，本脚本形同虚设 ---
    nondegenerate = float(delta_i.abs().min()) > 1e-3
    ok &= nondegenerate
    print(f"[2] non-degenerate     min|delta| = {float(delta_i.abs().min()):.3e}  "
          f"{'OK' if nondegenerate else 'FAIL: delta ~ 0, sign check is vacuous'}")

    # --- 判据 3：反证 —— 把 delta 取反（= 方向写反）后必须与原值不同 ---
    flipped_differs = not torch.allclose(delta_t, -delta_i, atol=1e-6)
    ok &= flipped_differs
    print(f"[3] negative control   -delta 可区分 = {flipped_differs}  "
          f"{'OK' if flipped_differs else 'FAIL: 方向反了也测不出，说明判据 2 没护住'}")

    print("\nRESULT:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
