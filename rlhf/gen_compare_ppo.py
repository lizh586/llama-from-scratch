"""P4-6：SFT vs PPO 生成对比（三列）。

复用 P1 的 `sft/gen_compare.py` 机器，只换模型表：
  - SFT            = sft_lr1e-05_B32_T256_S1782.pt
  - PPO β=0.1 lr1e-6 = 收口配置（drift 3e-4，预期空转）
  - PPO β=0.1 lr1e-5 = hacking 调查批（KL末 1.19，预期退化）
其余（render / encode / greedy+5×sampling / SEED）一律不动 ——
PPO actor ckpt 是裸 63 键（无 `model.` 前缀），与 SFT 同口径，`build_model` 直接读。
"""
import json
from pathlib import Path
import sys
import torch

LLM = Path(__file__).resolve().parents[1]     # .../code/llm
sys.path += [str(LLM/"llama"), str(LLM/"pipeline"), str(LLM/"sft")]

import sft_data  # noqa: E402
from llama_model import LLaMA  # noqa: E402
from data import stoi, itos, vocab_size, EOS_ID  # noqa: E402

SEED = 1337
N_SAMPLES = 5
MAX_LEN = 256
TAU = 1
TOP_P = 0
TOP_K = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from paths import ROOT  # noqa: E402
INPUT_PATH = str(ROOT/"data"/"sft"/"gen_prompts.json")
OUTPUT_PATH = str(ROOT/"data"/"runs"/"ppo_gen_compare.json")

RUNS = ROOT/"data"/"runs"
CKPT_PATH = {
    'sft':      str(RUNS/"sft_lr1e-05_B32_T256_S1782.pt"),
    'ppo_1e-6': str(RUNS/"ppo_S200_B4_K4_lra1e-06_beta0.1_s0_actor.pt"),
    'ppo_1e-5': str(RUNS/"ppo_S200_B4_K4_lra1e-05_beta0.1_s0_actor.pt"),
}


def build_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt["config"]
    model = LLaMA(vocab_size, cfg["d_model"], cfg["n_layers"],
                  cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
                  ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def encode_prompt(instruction, input_, output):
    s, n = sft_data.render(instruction, input_, output)
    ids = sft_data.encode(s)
    return ids[:n], n


def decode(ids, itos):
    out = []
    for i in ids:
        tok = itos[int(i)]
        if tok == '<eos>':
            break
        out.append(tok)
    return ''.join(out)


def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    torch.manual_seed(SEED)

    models = {name: build_model(path) for name, path in CKPT_PATH.items()}

    results = []
    for idx, p in enumerate(prompts):
        prompt_ids, n = encode_prompt(p['instruction'], p['input'], p['output'])
        entry = {
            'index': idx,
            'instruction': p['instruction'],
            'input': p['input'],
            'output': p['output'],
            'variant': p['variant'],
            'n': n,
            'results': {},
        }

        for name, model in models.items():
            entry['results'][name] = {'greedy': None, 'sampling': []}
            greedy = model.generate(prompt_ids, MAX_LEN, None, temperature=0)
            assert greedy[:n] == prompt_ids, f"[{idx}][{name}] generate 开头不是 prompt，切点不可信"
            entry['results'][name]['greedy'] = greedy[n:]
            for i in range(N_SAMPLES):
                torch.manual_seed(SEED+i)
                entry['results'][name]['sampling'].append(
                    model.generate(prompt_ids, MAX_LEN, None,
                                   temperature=TAU, top_k=TOP_K, top_p=TOP_P)[n:])

        results.append(entry)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f)
    print(f"wrote {OUTPUT_PATH}  ({len(results)} prompts × {len(models)} models)")


if __name__ == '__main__':
    main()
