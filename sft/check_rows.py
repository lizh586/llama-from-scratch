"""规格 §8 校验断言的落地：任何来源的行（手写夹具 / Alpaca）都要过这一关。

check_row() / check_labels() 供数据管线 import；直接运行 = 跑 8 条夹具 + 全量 Alpaca 过滤集。
"""
import sys
import json
from pathlib import Path

SFT = Path(__file__).resolve().parent
sys.path += [str(SFT), str(SFT.parent / "pipeline")]

from data import T_DEFAULT, EOS_ID
from sft_data import render, encode, build_input_ids, build_labels

FIXTURES = SFT / "fixtures.json"
ALPACA = SFT.parents[1] / "data" / "sft" / "alpaca_sft.json"


def spec_render(instruction, input_, output):
    """按规格 §2 原文独立拼串（不走 render），用于比对模板原文是否被改过。"""
    prompt = "### Instruction:\n" + instruction + "\n\n"
    if input_:
        prompt += "### Input:\n" + input_ + "\n\n"
    prompt += "### Response:\n"
    return prompt + output, len(prompt)


def check_row(instruction, input_, output, T=T_DEFAULT):
    s, n = render(instruction, input_, output)
    ids = build_input_ids(s)

    s_spec, n_spec = spec_render(instruction, input_, output)
    assert s == s_spec, "模板原文与规格 §2 不一致"
    assert n == n_spec, f"切点 n={n} != 规格拼串长度 {n_spec}"

    n_resp = s.count('### Response:\n')
    n_inp = s.count('### Input:')

    assert s[n:] == output, "切点划错 -> response 段错位"
    assert n_resp == 1, f"'### Response:\\n' 出现 {n_resp} 次（应为 1）"
    assert n_inp == (1 if input_ else 0), f"变体路由错：'### Input:' 出现 {n_inp} 次，input_ 是否非空={bool(input_)}"
    assert ids[n : n + len(output) + 1] == [*encode(output), EOS_ID], "response 段 id 对不上：<eos> 被摊成 5 个 id / 漏补 eos"
    assert len(ids) <= T, f"被截断：len(ids)={len(ids)} > T={T}"

    return n, len(ids)


def check_labels(instruction, input_, output, T=T_DEFAULT):
    """mask 侧：labels 与 y = ids[1:] 对齐、只 mask prompt 段、response + EOS 完整保留。"""
    s, n = render(instruction, input_, output)
    ids = build_input_ids(s)
    labels = build_labels(n, ids)
    tail = [*encode(output), EOS_ID]

    assert len(labels) == len(ids) - 1, f"len(labels)={len(labels)} != len(ids)-1={len(ids)-1}（要对齐 y = ids[1:]）"
    assert labels[-len(tail):] == tail, "response 段（含 EOS）没被完整保留在有标签的一侧"
    assert labels[:-len(tail)] == [-100] * (n - 1), f"prompt 段 mask 位数 != n-1={n - 1}"

    return labels


if __name__ == '__main__':
    fixtures = json.loads(FIXTURES.read_text(encoding='utf-8'))
    print(f"夹具 {len(fixtures)} 条 —— §8 五条断言 + 模板原文比对")
    branch = {"A": 0, "B": 0}
    for f in fixtures:
        n, L = check_row(f["instruction"], f["input"], f["output"])
        check_labels(f["instruction"], f["input"], f["output"])
        variant = "A" if f["input"] else "B"
        branch[variant] += 1
        print(f"  {f['grid']:8} 变体{variant}  n={n:>4}  len(ids)={L:>4}  <=T: {L <= T_DEFAULT}")
    print(f"  全部通过（含 labels 侧）。变体覆盖：A x {branch['A']}   B x {branch['B']}")

    rows = json.loads(ALPACA.read_text(encoding='utf-8'))
    print(f"Alpaca 过滤集 {len(rows)} 条 —— 全量过检查 + JSON 存的 n 与重算一致")
    for i, r in enumerate(rows):
        n, L = check_row(r["instruction"], r["input"], r["output"])
        check_labels(r["instruction"], r["input"], r["output"])
        assert r["n"] == n, f"row {i}: JSON 存的 n={r['n']} != 重算 {n}（模板改过？）"
    print(f"  {len(rows)} 条全部通过（n 字段与当前模板一致；labels 侧全过）")
