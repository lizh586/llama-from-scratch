"""补上 2×2 里缺的那一格：SFT 末模型在 shakespeare val 上打多少。

              shakespeare val          alpaca val (response 口径)
  S800        1.5108 (pretrain 自己的)   2.9280
  SFT 末      ← 缺这一格                1.5597

这一格回答的是「SFT 有没有把 pretrain 的字符能力毁掉」（遗忘），
**不是**「SFT 是否达到了 pretrain 的水平」—— 后者的两个数在语料难度不同的轴上，不可比。

两条自检把口径钉死：
  ① S800 在 shakespeare val 上必须复现 1.5108（分母 y.numel()，与 train.py 同一套）
  ② S800 / SFT 在 alpaca val 上必须复现 2.9280 / 1.5597（分母 labels != -100）
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SFT_DIR = Path(__file__).resolve().parent
REPO = SFT_DIR.parents[0]
sys.path += [str(SFT_DIR), str(REPO / "llama"), str(REPO / "pipeline")]

import sft_batch
from data import get_data, vocab_size
from llama_model import LLaMA

RUNS = REPO / "data" / "runs"
PRETRAIN_CKPT = RUNS / "lr0.0003_B32_T256_S800.pt"
SFT_CKPT = RUNS / "sft_lr1e-05_B32_T256_S1782.pt"


def last_val(tag):
    with open(RUNS / f"{tag}.json", encoding="utf-8") as f:
        h = json.load(f)["history"]
    return [p["val_loss"] for p in h if p["val_loss"] is not None][-1]


def first_val(tag):
    with open(RUNS / f"{tag}.json", encoding="utf-8") as f:
        h = json.load(f)["history"]
    return [p["val_loss"] for p in h if p["val_loss"] is not None][0]


@torch.no_grad()
def eval_pretrain_style(model, loader, device):
    """与 pretrain/train.py:evaluate 逐字一致：分母 y.numel()，无 ignore_index。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


@torch.no_grad()
def eval_response(model, loader, device):
    """与 sft/train_sft.py:evaluate 逐字一致：分母 labels != -100。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size), labels.reshape(-1),
            reduction="sum", ignore_index=-100,
        )
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()
    model.train()
    return total_loss / total_tokens


def build(ckpt_path, device):
    ckpt = torch.load(ckpt_path)
    cfg = ckpt["config"]
    model = LLaMA(
        vocab_size, cfg["d_model"], cfg["n_layers"],
        cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, sh_val_loader, _ = get_data(T=256, B=32)

    rows = sft_batch.make_rows()
    val_ds = sft_batch.SFTDataset(rows[int(len(rows) * 0.9):])
    alp_val_loader = sft_batch.make_sft_loader(val_ds, shuffle=False, drop_last=False)

    m_pre = build(PRETRAIN_CKPT, device)
    m_sft = build(SFT_CKPT, device)

    p_sh = eval_pretrain_style(m_pre, sh_val_loader, device)
    s_sh = eval_pretrain_style(m_sft, sh_val_loader, device)
    p_al = eval_response(m_pre, alp_val_loader, device)
    s_al = eval_response(m_sft, alp_val_loader, device)

    exp_p_sh = last_val("lr0.0003_B32_T256_S800")
    exp_p_al = first_val("sft_lr1e-05_B32_T256_S1782")
    exp_s_al = last_val("sft_lr1e-05_B32_T256_S1782")

    assert abs(p_sh - exp_p_sh) < 5e-4, f"S800@shakespeare {p_sh:.4f} 复现不出 {exp_p_sh:.4f}"
    assert abs(p_al - exp_p_al) < 5e-4, f"S800@alpaca {p_al:.4f} 复现不出 {exp_p_al:.4f}"
    assert abs(s_al - exp_s_al) < 5e-4, f"SFT@alpaca {s_al:.4f} 复现不出 {exp_s_al:.4f}"

    forget = s_sh - exp_p_sh

    print("                shakespeare val      alpaca val (response)")
    print(f"  S800          {p_sh:.4f}               {p_al:.4f}")
    print(f"  SFT 末        {s_sh:.4f}               {s_al:.4f}")
    print()
    print(f"S800 换语料 shakespeare->alpaca : {p_al - p_sh:+.4f}")
    print(f"SFT  换语料 shakespeare->alpaca : {s_al - s_sh:+.4f}")
    print(f"SFT 遗忘量（shakespeare）        : {forget:+.4f}")
    print(f"SFT 收益量（alpaca）             : {s_al - p_al:+.4f}")

    out = RUNS / "sft_cross_matrix.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "S800_shakespeare": p_sh, "S800_alpaca_response": p_al,
            "SFT_shakespeare": s_sh, "SFT_alpaca_response": s_al,
            "forget_shakespeare": forget, "gain_alpaca": s_al - p_al,
        }, f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
