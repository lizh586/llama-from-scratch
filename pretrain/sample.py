import sys
from pathlib import Path
import torch
import statistics
from run_tag import make_tag


REPO = Path(__file__).resolve().parents[1]
sys.path += [str(REPO / "pipeline"), str(REPO / "llama")]


from llama_model import LLaMA

def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.int64)

def decode(ids, itos):
    out = []
    for i in ids:
        tok = itos[int(i)]
        if tok == '<eos>':
            break
        out.append(tok)
    return ''.join(out)

def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(n_grams)) / len(n_grams)

def report(text):
    words = text.split()
    print(text)
    print(f"words={len(words):3d}"
            f"  word_d2={distinct_n(words,2):.3f} (denom={len(words)-1})"
            f"  char_d2={distinct_n(text,2):.3f} (denom={len(text)-1})"
            f"  word_d3={distinct_n(words,3):.3f} (denom={len(words)-2})"
            f"  char_d3={distinct_n(text,3):.3f} (denom={len(text)-2})")
    


# ----------------- 常量 --------------------


CKPT_TAG = "lr0.0003_B32_T256_S800.pt"
CKPT_PATH = REPO / "data" / "runs" / CKPT_TAG
ckpt = torch.load(CKPT_PATH)

cfg = ckpt["config"]
expected_tag = make_tag(cfg) + ".pt"
assert CKPT_TAG == expected_tag, f"权重与 config 不符: {CKPT_TAG} != {expected_tag}"
vocab_size = ckpt["vocab_size"]
max_len = 200


model = LLaMA(
    vocab_size, cfg["d_model"], cfg["n_layers"],
    cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
    )
model.load_state_dict(ckpt["state_dict"])

stoi = ckpt["stoi"]
itos = ckpt["itos"]


# ------------------ sweep --------------------
prompt = "ROMEO:"
ids = encode(prompt, stoi).tolist()
N = 5
CONFIGS = [
    ("greedy", dict(temperature=0, top_k=0, top_p=0.0)),
    ("ctrl", dict(temperature=1.0, top_k=0, top_p=0.0)),
    ("tau=0.5", dict(temperature=0.5, top_k=0, top_p=0.0)),
    ("tau=0.8", dict(temperature=0.8, top_k=0, top_p=0.0)),
    ("tau=1.2", dict(temperature=1.2, top_k=0, top_p=0.0)),
    ("k=5", dict(temperature=1.0, top_k=5, top_p=0.0)),
    ("k=10", dict(temperature=1.0, top_k=10, top_p=0.0)),
    ("k=20", dict(temperature=1.0, top_k=20, top_p=0.0)),
    ("p=0.9",   dict(temperature=1.0, top_k=0,  top_p=0.9)),
    ("p=0.95",  dict(temperature=1.0, top_k=0,  top_p=0.95)),    
]

results = {}

for name, params in CONFIGS:
    texts = []
    if params["temperature"] == 0:
        # greedy 只有一条
        out = model.generate(ids, max_len, None, **params)
        texts.append(decode(out, itos))
    else:
        for seed in range(N):
            torch.manual_seed(seed)     # 按样本序号播种，跨配置一致
            out = model.generate(ids, max_len, None, **params)
            texts.append(decode(out, itos))
    results[name] = texts

    # 每个配置留一条打印
    print(f"\n{name}")
    report(texts[0])

# 汇总表
for name, texts in results.items():
    d2s = [distinct_n(t, 2) for t in texts]
    m2 = sum(d2s)/len(d2s)
    s2 = statistics.stdev(d2s) if len(d2s) > 1 else 0.0
    print(f"{name:<10} {len(texts):>3} "
          f"  {m2:.3f}±{s2:.3f}     ")

# 精确相等测试
print("equal-to-ctrl prob (sample 0, exact string match)")

ref = results["ctrl"][0]
for name, texts in results.items():
    same = (texts[0] == ref)
    print(f"{name} == ctrl[0]: {same}")

# ctrl 自己 5 条之间有没有重复
print("\nctrl internal duplicates (should be 0)")
dups = 0
for i in range(len(results["ctrl"])):
    for j in range(i+1, len(results["ctrl"])):
        if results["ctrl"][i] == results["ctrl"][j]:
            print(f" ctrl[{i}] == ctrl[{j}]")
            dups += 1
print(f"total: {dups}")




