"""ppo_gen_compare.json 的断言（只读 JSON，不加载模型）。

常量在这里独立重述冻结值，不从 gen_compare_ppo.py import
—— 检查器对着契约查，不跟着生产者一起改。

判据② 扩成三列：报 sft↔ppo_1e-6 / sft↔ppo_1e-5 两对的可区分条数；
硬断言只落在 lr=1e-5 那对（预期：收口 lr=1e-6 空转、lr=1e-5 可区分）。
"""
import json
import sys
from pathlib import Path

LLM = Path(__file__).resolve().parents[1]          # .../code/llm
sys.path += [str(LLM / "pipeline")]
from data import stoi, itos, EOS_ID, vocab_size

from paths import ROOT  # noqa: E402
PATH = ROOT / "data" / "runs" / "ppo_gen_compare.json"

N_PROMPTS = 10          # 冻结：gen_prompts.json 条数
N_SAMPLES = 5           # 冻结：sampling 重复数
MAX_LEN = 256           # 冻结：全链序列长度
MODELS = ('sft', 'ppo_1e-6', 'ppo_1e-5')
PAIRS = (('sft', 'ppo_1e-6'), ('sft', 'ppo_1e-5'))
HARD_PAIR = ('sft', 'ppo_1e-5')     # 预期可区分的那对
MIN_DISTINCT = 5                    # 判据② 门槛（同 P1）
HASH_ID = [stoi[c] for c in '###']


def has_subseq(seq, sub):
    m = len(sub)
    return any(seq[i:i + m] == sub for i in range(len(seq) - m + 1))


def eff_len(seq):
    """首个 EOS 的位置即有效长度；无 EOS 则满长。"""
    return seq.index(EOS_ID) if EOS_ID in seq else len(seq)


def main():
    if not PATH.exists():
        raise SystemExit(f"{PATH} 不存在 —— 先跑 gen_compare_ppo.py")

    rows = json.loads(PATH.read_text(encoding='utf-8'))

    assert len(rows) == N_PROMPTS, f"条数 {len(rows)} != {N_PROMPTS}"
    assert [r['index'] for r in rows] == list(range(N_PROMPTS)), "index 不是 0..9"

    for r in rows:
        i, n = r['index'], r['n']
        assert r['variant'] == ('A' if r['input'] == '' else 'B'), f"[{i}] variant 与 input 不一致"
        assert set(r['results']) == set(MODELS), f"[{i}] 模型键 {sorted(r['results'])}"

        for name in MODELS:
            res = r['results'][name]
            g, s = res['greedy'], res['sampling']

            assert isinstance(g, list) and g and isinstance(g[0], int), \
                f"[{i}][{name}] greedy 不是 ids 列表：{type(g).__name__}"
            assert len(g) == MAX_LEN - n, \
                f"[{i}][{name}] greedy 长度 {len(g)} != {MAX_LEN - n}（切点或 max_len 错）"
            assert len(s) == N_SAMPLES, f"[{i}][{name}] sampling {len(s)} 条 != {N_SAMPLES}"

            for k, seq in enumerate(s):
                assert isinstance(seq, list) and seq and isinstance(seq[0], int), \
                    f"[{i}][{name}] sampling[{k}] 不是 ids 列表"
                assert len(seq) == MAX_LEN - n, \
                    f"[{i}][{name}] sampling[{k}] 长度 {len(seq)} != {MAX_LEN - n}"

            for tag, seq in [('greedy', g)] + [(f'sampling[{k}]', x) for k, x in enumerate(s)]:
                bad = [t for t in seq if not (0 <= t < vocab_size)]
                assert not bad, f"[{i}][{name}] {tag} 越界 id: {bad[:5]}"

    # ---- 判据②：greedy 可区分条数（逐对报，硬断言落在 HARD_PAIR）----
    counts = {}
    for a, b in PAIRS:
        diff = [r['index'] for r in rows
                if r['results'][a]['greedy'] != r['results'][b]['greedy']]
        counts[(a, b)] = len(diff)
        print(f"greedy {a:8s} vs {b:8s} 不同：{len(diff):2d}/{N_PROMPTS}  {diff}")
    assert counts[HARD_PAIR] >= MIN_DISTINCT, \
        f"{HARD_PAIR} 可区分只 {counts[HARD_PAIR]} 条 < {MIN_DISTINCT}"

    # ---- 代理读数（测量值，不断言）----
    for name in MODELS:
        seqs = [seq for r in rows
                for seq in [r['results'][name]['greedy']] + r['results'][name]['sampling']]
        eos = sum(EOS_ID in seq for seq in seqs)
        hsh = sum(has_subseq(seq, HASH_ID) for seq in seqs)
        mlen = sum(eff_len(seq) for seq in seqs) / len(seqs)
        print(f"  {name:9s} 段数={len(seqs):2d}  含 EOS {eos:2d}/{len(seqs)}  "
              f"含 '###' {hsh:2d}/{len(seqs)}  有效长度均值 {mlen:6.1f}")

    print("OK")


if __name__ == '__main__':
    main()
