"""
从本地缓存重跑 Alpaca SFT 过滤，覆盖写：
  data/sft/alpaca_sft.json

字段：instruction / input / output / n
n 取自 sft_data.render 的切点。
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset
import sft_data

VOCAB = sft_data.VOCAB
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUT = ROOT / "data" / "sft" / "alpaca_sft.json"

def main():
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    total = len(ds)

    empty = 0
    non_vocab = 0
    too_long = 0
    rows=[]
    fn = sft_data.render
    for ex in ds:
        output = ex["output"] or ""

        if output.strip() == "":
            empty+=1
            continue

        s, n = fn(ex["instruction"],ex["input"],ex["output"])
        if any(ch not in VOCAB for ch in s):
            non_vocab += 1
            continue

        if len(s) + 1 > 256:
            too_long += 1
            continue

        rows.append({
            "instruction":ex["instruction"],
            "input":ex["input"],
            "output":ex["output"],
            "n":n,
        })
    print(f"total={total}  empty={empty}  non_ascii={non_vocab}  too_long={too_long}  kept={len(rows)}")
    print(f"non_ascii {non_vocab/total:.2%}  too_long {too_long/total:.2%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(rows, f)


if __name__ == '__main__':
    main()

