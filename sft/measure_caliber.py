"""把 1.5108 与 2.9280 之间的 1.4173 拆成「评测集差」和「口径差」两个可单独测的量。

缺的那个数是 L_all^alpaca —— S800 在 alpaca val 上、用全位置分母（prompt+response）打的分。
补上它之后：

    口径差   = 2.9280 - L_all^alpaca     （同一份 alpaca val，换分母）
    评测集差 = L_all^alpaca - 1.5108     （同为全位置分母，换评测集 shakespeare→alpaca）
    两者之和必须 = 1.4173                ← 恒等式，这条不成立说明测量写错了

同一次 forward 里算两个口径，省一半时间。
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SFT = Path(__file__).resolve().parent
REPO = SFT.parents[0]
sys.path += [str(SFT), str(REPO / "llama"), str(REPO / "pipeline")]

import sft_batch
from data import PAD_ID, EOS_ID, vocab_size
from llama_model import LLaMA

CKPT_TAG = "lr0.0003_B32_T256_S800.pt"
CKPT_PATH = REPO / "data" / "runs" / CKPT_TAG
RUNS = REPO / "data" / "runs"


def last_val(tag):
    with open(RUNS / f"{tag}.json", encoding="utf-8") as f:
        h = json.load(f)["history"]
    return [p["val_loss"] for p in h if p["val_loss"] is not None][-1]


def first_val(tag):
    with open(RUNS / f"{tag}.json", encoding="utf-8") as f:
        h = json.load(f)["history"]
    return [p["val_loss"] for p in h if p["val_loss"] is not None][0]


# 两个被拆开的分，都从 run JSON 全精度取 —— 不写四舍五入值，否则 checksum 差 1e-4 说不清是谁的锅
PRETRAIN_ALLVAL = last_val("lr0.0003_B32_T256_S800")            # 全位置口径 × shakespeare val
SFT_STEP0 = first_val("sft_lr1e-05_B32_T256_S1782")             # response 口径 × alpaca val
GAP = SFT_STEP0 - PRETRAIN_ALLVAL


def real_len(x):
    """每行真实前缀长度 L。x = ids + [PAD]*(T-len(ids))，且 ids 末尾是 EOS。

    PAD_ID == EOS_ID，所以值相等判不出 pad 从哪开始；但这个位置**必然结尾全是 PAD**
    （真实段中间不会出现 96）。取「从这里往后全 True」的最早位置，就是末位 EOS 所在处 L-1。
    """
    is_pad = x == PAD_ID                                    # (B, T)
    tail = is_pad.flip(1).cumprod(1).flip(1).bool()         # True 当且仅当 i..T-1 全是 PAD
    return tail.float().argmax(1) + 1                       # (B,)


@torch.no_grad()
def measure(model, loader, device):
    model.eval()
    all_sum = 0.0
    all_n = 0
    resp_sum = 0.0
    resp_n = 0
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        B_, T_ = x.shape
        logits = model(x)

        # ---- 全位置口径：位置 t 预测 x[t+1]，把每个真实位置都算上（prompt + response）----
        L = real_len(x)                                                  # (B,)
        keep = torch.arange(T_ - 1, device=device)[None, :] < (L - 1)[:, None]
        per_pos = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            x[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(B_, T_ - 1)
        all_sum += per_pos[keep].sum().item()
        all_n += int(keep.sum())

        # ---- response 口径：和主训练 evaluate() 完全同一套（分母 = labels != -100）----
        r = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        resp_sum += r.item()
        resp_n += int((labels != -100).sum())

    model.train()
    return all_sum / all_n, all_n, resp_sum / resp_n, resp_n


def main():
    ckpt = torch.load(CKPT_PATH)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = sft_batch.make_rows()
    split = int(len(rows) * 0.9)
    val_ds = sft_batch.SFTDataset(rows[split:])
    val_loader = sft_batch.make_sft_loader(val_ds, shuffle=False, drop_last=False)

    model = LLaMA(
        vocab_size, cfg["d_model"], cfg["n_layers"],
        cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    l_all, n_all, l_resp, n_resp = measure(model, val_loader, device)

    # ---- 自检 1：response 口径必须复现主训练的 step-0 基线 2.9280 ----
    assert n_resp == 121167, f"response 分母 {n_resp} != 121167（主训练实测值）"
    assert abs(l_resp - SFT_STEP0) < 5e-4, f"response 口径 {l_resp:.4f} 复现不出 {SFT_STEP0}"

    # ---- 自检 2：全位置分母必须等于 P+R ----
    n_prompt = n_all - n_resp
    assert n_all == 361034, f"全位置分母 {n_all} != 361034"

    # ---- 自检 3：分解的两个分量之和必须等于原缺口 ----
    d_caliber = l_resp - l_all
    d_corpus = l_all - PRETRAIN_ALLVAL
    assert abs(d_caliber + d_corpus - GAP) < 1e-9, (
        f"分解不闭合：(口径差 {d_caliber:.6f}) + (评测集差 {d_corpus:.6f})"
        f" = {d_caliber + d_corpus:.6f} != {GAP:.6f}"
    )

    # prompt 段的平均 loss 由加权恒等式反解：它是唯一没直接测的量
    l_prompt = (l_all * n_all - l_resp * n_resp) / n_prompt

    print(f"S800 @ alpaca val（{len(val_ds)} 条）")
    print(f"  全位置口径  L_all^alpaca = {l_all:.4f}  分母 {n_all}"
          f"  (prompt {n_prompt} + response {n_resp})")
    print(f"  response 口径          = {l_resp:.4f}  分母 {n_resp}   <- 对比主训练 {SFT_STEP0}")
    print(f"  反解 prompt 口径       = {l_prompt:.4f}  分母 {n_prompt}")
    print()
    print(f"1.4173 = 口径差 {d_caliber:.4f} + 评测集差 {d_corpus:.4f}"
          f"  (checksum {d_caliber + d_corpus:.4f})")
    print(f"  └ 口径差那一半的意思是：alpaca val 上 response 段比 prompt 段难 {l_resp - l_prompt:.4f}")

    out = REPO / "data" / "runs" / "sft_caliber_decomposition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "ckpt": CKPT_TAG,
            "alpaca_val_rows": len(val_ds),
            "L_all_alpaca": l_all, "n_all": n_all,
            "n_prompt": n_prompt, "n_response": n_resp,
            "L_response": l_resp, "L_prompt_solved": l_prompt,
            "L_all_shakespeare": PRETRAIN_ALLVAL,
            "delta_caliber": d_caliber, "delta_corpus": d_corpus, "gap": GAP,
        }, f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
