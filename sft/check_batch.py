"""校验 sft_batch.build_tensor_data：形状 / dtype / pad 区 / shift 不变式。

核心不变式只有一条 —— **labels[t] 必须等于 x[t+1]**（非 -100 处）。
它是 batch 层唯一能证明「没和 pad 串位」的判据：补长时 x 补 k 个、
labels 补 k+1 个，两者一旦互相借 len，这条必炸。
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sft_batch
import sft_data
from data import EOS_ID

T = sft_batch.T
PAD_ID = sft_batch.PAD_ID
REPO = Path(__file__).resolve().parents[1]


def verify(x, labels, n, L, where):
    """L = len(ids)（补长前）。"""
    assert x.shape == (T,), f"{where} x.shape={tuple(x.shape)} != ({T},)"
    assert labels.shape == (T,), f"{where} labels.shape={tuple(labels.shape)} != ({T},)"
    assert x.dtype == torch.int64 and labels.dtype == torch.int64, f"{where} dtype"

    # ① pad 区 = [L, T)，必须是连续后缀、且全是 PAD_ID
    assert bool((x[L:] == PAD_ID).all()), f"{where} x[{L}:] 不全是 PAD_ID"
    assert bool((x[: L - 1] != PAD_ID).all()), f"{where} 真实段内出现了 PAD_ID"
    # PAD_ID == EOS_ID，所以 x[L-1] 的值与 pad 不可区分 —— 这是值域上的事实，
    # 不是 bug：P3 左 pad 必须用位置累积 mask，不能靠 x == PAD_ID 判。
    assert int(x[L - 1]) == EOS_ID, f"{where} 末尾不是 EOS"

    # ② labels 的 pad / mask 区必须覆盖 [0, n-1) ∪ [L-1, T)
    assert bool((labels[: n - 1] == -100).all()), f"{where} prompt 段没被完整 mask"
    assert bool((labels[L - 1 :] == -100).all()), f"{where} EOS 位/补长位应为 -100"
    assert int((labels != -100).sum()) == L - n, f"{where} response 有效位 != L-n"

    # ③ shift 不变式：labels[t] == x[t+1]（只对非 -100 的位）
    idx = (labels != -100).nonzero().flatten()
    assert torch.equal(labels[idx], x[idx + 1]), f"{where} labels 与 x 错位了"


def main():
    rows = json.loads((REPO / "data" / "sft" / "alpaca_sft.json").read_text(encoding="utf-8"))
    fixtures = json.loads((REPO / "sft" / "fixtures.json").read_text(encoding="utf-8"))
    print(f"夹具 {len(fixtures)} 条 · Alpaca {len(rows)} 条")

    maxL = 0
    for i, r in enumerate(fixtures + rows):
        s, n = sft_data.render(r["instruction"], r["input"], r["output"])
        L = len(s) + 1
        maxL = max(maxL, L)
        assert L <= T, f"row {i} 长度 {L} > T，补长会静默失效"
        x, labels = sft_batch.build_tensor_data(r)
        verify(x, labels, n, L, f"row {i}")

    print(f"全过 · 最长行 L = {maxL} / T = {T}")

    # ---- make_rows() 必须指向过滤集本身 ----
    # 指向 data/alpaca/alpaca.jsonl（52002 条原始）时这条会炸：
    # 原始集里 OOV 会 KeyError，超长会因 T-len 为负静默不补。
    rows_m = sft_batch.make_rows()
    assert rows_m == rows, f"make_rows() 出的不是过滤集：{len(rows_m)} 条 vs 期望 {len(rows)}"

    # ---- DataLoader 层 ----
    B = sft_batch.B
    loader = sft_batch.make_sft_loader(sft_batch.SFTDataset(rows_m))
    n_batches = len(loader)
    dropped = len(rows_m) - n_batches * B
    assert n_batches == len(rows_m) // B, f"批数 {n_batches} != {len(rows_m)//B}"
    assert dropped == 2, f"drop_last 应丢 2 条，实际 {dropped}"

    xb, lb = next(iter(loader))
    assert xb.shape == (B, T), f"batch x.shape={tuple(xb.shape)} != ({B},{T})"
    assert lb.shape == (B, T), f"batch labels.shape={tuple(lb.shape)} != ({B},{T})"
    assert xb.dtype == torch.int64 and lb.dtype == torch.int64, "batch dtype"

    bi, si = (lb != -100).nonzero().T
    assert int(si.max()) < T - 1, "有效位落到了最后一位 —— EOS 位没被 mask"
    assert torch.equal(lb[bi, si], xb[bi, si + 1]), "batch 层 shift 不变式破裂"

    print(f"loader 过 · {n_batches} 批 × B={B} = {n_batches*B} 条（丢 {dropped}）· batch shift 不变式成立")

    # ---- 变异反证：漏掉 shift（labels 取 ids[n-1:] 而非 ids[n:]），③ 必须炸 ----
    r = fixtures[3]
    s, n = sft_data.render(r["instruction"], r["input"], r["output"])
    ids = sft_data.build_input_ids(s)
    L = len(ids)
    bad = [-100] * (n - 1) + ids[n - 1 : L - 1] + [-100] * (T - L + 1)
    assert len(bad) == T, len(bad)
    bad_t = torch.tensor(bad, dtype=torch.int64)
    x, _ = sft_batch.build_tensor_data(r)
    try:
        verify(x, bad_t, n, L, "变异")
    except AssertionError as e:
        print(f"变异被拦 -> {e}")
    else:
        raise SystemExit("变异没被拦 —— 这条检查是假的")


if __name__ == "__main__":
    main()
