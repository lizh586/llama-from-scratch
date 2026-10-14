import random
import json
from pathlib import Path

from paths import ROOT  # noqa: E402

WORDS = [
    # len 3
    "cat", "dog", "sun", "red", "box", "car", "map", "pen", "cup", "key",
    # len 4
    "blue", "fish", "book", "tree", "moon", "star", "bird", "door", "lamp", "desk",
    # len 5
    "apple", "happy", "green", "house", "light", "water", "music", "paper", "table", "chair",
    # len 6
    "orange", "purple", "yellow", "silver", "flower", "garden", "summer", "winter", "mother", "father",
    # len 7
    "country", "picture", "problem", "student", "brother", "morning", "evening", "teacher", "library", "machine",
    # len 8
    "mountain", "computer", "building", "keyboard", "elephant", "anything", "baseball", "birthday", "daughter", "medicine",
    # len 9
    "beautiful", "wonderful", "important", "different", "knowledge", "something", "sometimes", "afternoon", "chocolate", "vegetable",
    # len 10
    "technology", "university", "government", "experience", "strawberry", "everything", "understand", "commercial", "background", "production"
]
SEED = 0
# 2026-10-01：8 → 40 条。8 族 × 5 条；训练用每族的 {0,4} 号、held-out 用 {1,2,3} 号
# ⇒ 两侧各 10 族位，前 4 族「明说规则」/ 后 4 族「泛指任务」各 10，零交集。
#
# ⚠️ 规则 v3 起，明说组必须说的是**第一个**词，不能说 last —— 否则标签语义与指令相反。
#    前 4 族（explicit_instructions）由 SRC/WRAP 程序化生成，保证「规则口径」单一来源。
_explicit_src = [
    "Copy the first word of the input to the end of your response.",
    "Repeat the opening word of the input.",
    "Output the first word you see in the input.",
    "What is the first word of the input? Answer with it.",
    "End your answer with the word the input begins with.",
    "Append the input's first word to your response.",
    "Put the input's opening word at the end of your reply.",
    "Which word starts the input? Write it last.",
    "Finish your response with the first word of the input.",
    "Take the input's first word and write it at the end.",
]
_explicit_wrap = [
    "Make sure the input's first word appears at the end.",
    "The input's opening word must end your answer.",
    "Conclude with the first word of the input.",
    "Your response should finish with the input's first word.",
    "Write the input's first word as your final word.",
    "Start from the input's first word and end your answer with it.",
    "Identify the input's first word and put it at the end.",
    "Answer using the first word of the input, placed last.",
    "Echo the input's first word when you finish.",
    "Locate the input's first word and repeat it at the end.",
]
_vague_a = [
    "Answer the question below.",
    "Do what is asked below.",
    "Read the request and reply.",
    "Complete the task.",
    "Provide a response to the input.",
    "Respond to the input.",
    "Carry out the instruction that follows.",
    "Handle the request that follows.",
    "Read the input and write a short response.",
    "Give a brief answer to the input.",
]
_vague_b = [
    "Please process the following input as instructed.",
    "Kindly answer the request below.",
    "Follow the instruction below carefully.",
    "Read the following and give a brief answer.",
    "Please follow the directions in the input.",
    "Here is a task. Carry it out.",
    "The task is below. Please do it.",
    "Follow the instruction that comes next.",
    "Read the task below and respond.",
    "Process the input according to the instruction.",
]


def _assign(cands, k_per_group, n_groups=2):
    """把 10 个候选均分到 n_groups 组（每组长 k_per_group=5），确定性交错分配。"""
    groups = [[] for _ in range(n_groups)]
    for i, s in enumerate(cands):
        groups[i % n_groups].append(s)
    assert all(len(g) == k_per_group for g in groups), [len(g) for g in groups]
    return groups


_g1, _g4 = _assign(_explicit_src, 5)     # 族1 / 族4
_g2, _g3 = _assign(_explicit_wrap, 5)    # 族2 / 族3
_g5, _g8 = _assign(_vague_a, 5)          # 族5 / 族8
_g6, _g7 = _assign(_vague_b, 5)          # 族6 / 族7

INSTRUCTIONS = _g1 + _g2 + _g3 + _g4 + _g5 + _g6 + _g7 + _g8
assert len(INSTRUCTIONS) == 40 and len(set(INSTRUCTIONS)) == 40

def make_body():
    k = random.randint(3, 8)
    words = random.sample(WORDS, k)
    return " ".join(words)

def make_pair(instruction):
    while True:
        # 1. Pick W: the last word of input
        W = random.choice(WORDS)

        # 2. Build input ending with W
        other_words = [w for w in WORDS if w != W]
        n_prefix = random.randint(3, 8)
        prefix_words = random.sample(other_words, n_prefix)
        input_text = " ".join(prefix_words) + " "+ W

        # 3. Pick W2: rejected ending word, not W, not ending with W
        candidates = [w for w in WORDS if w != W and not w.endswith(W)]
        W2 = random.choice(candidates)

        # 4. Build responses 

        chosen = make_body() + " " + W
        rejected = make_body() + " " + W2

        if len(instruction) + len(input_text) + len(chosen) > 209:
            continue
        if len(instruction) + len(input_text) + len(rejected) > 209:
            continue
        return {
            "instruction":instruction,
            "input":    input_text,
            "chosen":   chosen,
            "rejected": rejected,
        }

def generate_round_robin(n, instructions):
    pairs = []
    for i in range(n):
        instr = instructions[i % len(instructions)]
        pairs.append(make_pair(instr))
    return pairs

# ---------------- 规则 v3（2026-10-01）----------------
# v1/v2 失败根因：规则要 head 比较「response 末词」与「input 末词」两个词身份，
# 而 input 末词的位置 = len(prompt)+len(input)-|W|，**每条都不同** ⇒ 固定位置的线性读出取不到。
# v3 把目标词挪到 **input 的第一个词**：偏移固定、与 input 长度无关，末位 EOS 一跳可取。

TARGET_POS = 0

def input_words(input_text):
    return input_text.split()

def target_word(input_text):
    return input_words(input_text)[TARGET_POS]

def make_pair_v3(instruction):
    while True:
        W = random.choice(WORDS)                     # 目标词 = input[0]
        others = [w for w in WORDS if w != W]
        mid = random.sample(others, random.randint(3, 8))
        last = random.choice(others)                 # 不能是 W：否则 W 同时是 input 末词
        input_text = " ".join([W] + mid + [last])    # 末词 != W ⇒ 不能靠"末词"位置取胜

        assert input_words(input_text)[0] == W and input_words(input_text)[-1] != W
        W2 = random.choice([w for w in WORDS if w != W])   # rejected 末词 != 目标词

        chosen = make_body() + " " + W
        rejected = make_body() + " " + W2

        # 额外捷径堵死：chosen 不能恰好以 input 末词结尾（否则旧规则也能拿分）
        if chosen.endswith(last) or rejected.endswith(W):
            continue
        if len(instruction) + len(input_text) + len(chosen) > 209:
            continue
        if len(instruction) + len(input_text) + len(rejected) > 209:
            continue
        return {
            "instruction":instruction,
            "input":    input_text,
            "chosen":   chosen,
            "rejected": rejected,
        }

def generate_round_robin_v3(n, instructions):
    pairs = []
    for i in range(n):
        pairs.append(make_pair_v3(instructions[i % len(instructions)]))
    return pairs

RULE_TEXT = {
    "v2": "chosen 的 response 末尾复述了 input 的最后一个非空单词，rejected 的末尾不是该单词。判定依据 response.endswith(input_last_word)。",
    "v3": "chosen 的 response 末尾复述了 input 的**第一个**词（固定偏移），rejected 的末尾不是该词。判定依据 response.endswith(input_words[0])。",
}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["v2", "v3"], default="v3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(SEED)
    # 按 instruction 切分（A 方案），不按 pair 切。40 条 → 两侧各 20 条，零交集。
    # 每族 5 条，训练取 {0,4}、held-out 取 {1,2,3} ⇒ 两侧的族分布与明说/泛指比例对称。
    train_idx = [f * 5 + k for f in range(8) for k in (0, 4)]        # 16 条
    heldout_idx = [f * 5 + k for f in range(8) for k in (1, 2, 3)]   # 24 条
    train_instr = [INSTRUCTIONS[i] for i in train_idx]
    heldout_instr = [INSTRUCTIONS[i] for i in heldout_idx]
    assert not (set(train_instr) & set(heldout_instr))

    n_train = 160
    n_heldout = 48
    gen = generate_round_robin_v3 if args.rule == "v3" else generate_round_robin
    train = gen(n_train, train_instr)
    heldout = gen(n_heldout, heldout_instr)
    data = {
        "rule": RULE_TEXT[args.rule],
        "rule_version": args.rule,
        "seed": SEED,
        "caliber": {
            "p1": 0.65,
            "z": 1.645,
            "side": "one",
            "n_heldout": n_heldout,
            "n_total": n_train + n_heldout,
            "note": "one-sided alpha=0.05; n_min for p1=0.65",
            "instr_pool": len(INSTRUCTIONS),
            "instr_train": len(train_instr),
            "instr_heldout": len(heldout_instr),
            "split": "by_instruction",
            "target_pos": TARGET_POS if args.rule == "v3" else None,
        },
        "train": train,
        "heldout": heldout
    }    
    out = Path(args.out) if args.out else \
        ROOT / "data" / "rlhf" / "toy_pairs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{out}  rule={args.rule}  train={len(train)} heldout={len(heldout)} "
          f"instr={len(train_instr)}/{len(heldout_instr)}")
    

if __name__ == "__main__":
    main()