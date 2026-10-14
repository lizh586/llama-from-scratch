"""RM ckpt 判据 —— 键格式 + pooling 盖章 + 自包含 round-trip + 分派的反证。

规则（9/28 冻结，方案 A）：ckpt 的 state_dict 必须能 load 回产生它的对象。
  SFT ckpt -> 裸 LLaMA 键（embedding.weight ...）
  RM  ckpt -> 带前缀键（model.* 62 个 + scalar_head.*）
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from paths import ROOT  # noqa: E402
sys.path.insert(0, str(HERE))

from rm import RM  # noqa: E402
from rm_batch import RM_Dateset, make_rm_loader  # noqa: E402
from train_rm import compute_delta  # noqa: E402
from data import stoi, itos, EOS_ID, vocab_size  # noqa: E402
from llama_model import LLaMA  # noqa: E402

RUNS = ROOT / "data" / "runs"
SFT_CKPT = RUNS / "sft_lr1e-05_B32_T256_S1782.pt"
RM_CKPT = RUNS / "rm_lr1e-04_B4_T256_S50.pt"
TMP_CKPT = RUNS / "_roundtrip_tmp.pt"

PAIRS = [
    {"instruction": "Say hello.", "input": "", "chosen": "Hello!", "rejected": "Goodbye."},
    {"instruction": "Say hello.", "input": "", "chosen": "Hi there, nice to meet you.", "rejected": "No."},
    {"instruction": "Reverse the string.", "input": "abc", "chosen": "cba", "rejected": "abc"},
    {"instruction": "Add the numbers.", "input": "1 2", "chosen": "3", "rejected": "12"},
]


def wrap(model):
    """与 train_rm.py 相同的 6 字段格式。"""
    return {
        "config":     model.config,
        "vocab_size": vocab_size,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "stoi":       stoi,
        "itos":       itos,
        "eos_id":     EOS_ID,
    }


def bare_llama(cfg):
    return LLaMA(vocab_size, cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                 cfg["n_kv_heads"], cfg["head_dim"])


def main():
    ok = True
    torch.manual_seed(0)
    x, m = next(iter(make_rm_loader(RM_Dateset(PAIRS), shuffle=False)))

    # ============ 1. 键格式 ============
    ck = torch.load(RM_CKPT, map_location="cpu")
    keys_ok = set(ck) == {"config", "vocab_size", "state_dict", "stoi", "itos", "eos_id"}
    print(f"[1] 6 fields           got {sorted(ck)}  {'OK' if keys_ok else 'FAIL'}")
    ok &= keys_ok

    sd = ck["state_dict"]
    # 裸 LLaMA 的键数从 SFT ckpt 数出来，不硬编码
    n_bare_ref = len(torch.load(SFT_CKPT, map_location="cpu")["state_dict"])
    n_scalar = sum(k.startswith("scalar_head.") for k in sd)
    n_pref = sum(k.startswith("model.") for k in sd)
    n_bare = sum(not k.startswith(("model.", "scalar_head.")) for k in sd)
    fmt_ok = n_scalar == 1 and n_pref == n_bare_ref and n_bare == 0
    print(f"[2] key namespace      scalar_head.*={n_scalar}  model.*={n_pref}"
          f" (bare ref={n_bare_ref})  bare={n_bare}  {'OK' if fmt_ok else 'FAIL'}")
    ok &= fmt_ok

    # ============ 2. pooling 盖章 ============
    pool_ok = ck["config"].get("pooling") == "last"
    print(f"[3] config['pooling']  got {ck['config'].get('pooling')!r}  {'OK' if pool_ok else 'FAIL'}")
    ok &= pool_ok

    # ============ 3. 真实产物能装回去（分派走 if 分支） ============
    model = RM(str(RM_CKPT))
    d_real = compute_delta(model, x, m)
    finite = bool(torch.isfinite(d_real).all())
    sane = finite and float(d_real.detach().abs().min()) > 1e-3
    print(f"[4] load trained ckpt  delta={d_real.tolist()}  {'OK' if sane else 'FAIL'}")
    ok &= sane

    # ============ 4. 自包含 round-trip：RM(sft) -> 存 -> RM(存) -> 逐位相同 ============
    try:
        a = RM(str(SFT_CKPT))
        d_a = compute_delta(a, x, m)
        torch.save(wrap(a), TMP_CKPT)
        b = RM(str(TMP_CKPT))
        d_b = compute_delta(b, x, m)
        rt = torch.equal(d_a, d_b)
        print(f"[5] round-trip         max|diff| = {float((d_a-d_b).abs().max()):.3e}  exact={rt}  "
              f"{'OK' if rt else 'FAIL'}")
        ok &= rt
    finally:
        TMP_CKPT.unlink(missing_ok=True)

    # ============ 5. 反证：RM 的前缀键灌进裸 LLaMA 必须仍然报错 ============
    try:
        bare_llama(ck["config"]).load_state_dict(sd)
        raised = False
    except RuntimeError:
        raised = True
    print(f"[6] negative control   bare LLaMA 拒绝前缀键 = {raised}  "
          f"{'OK' if raised else 'FAIL: 两个命名空间一样，分派是白写的'}")
    ok &= raised

    print("\nRESULT:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
