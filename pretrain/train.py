import sys
from pathlib import Path
import json
import time
from run_tag import make_tag

REPO = Path(__file__).resolve().parents[1]
sys.path += [str(REPO / "pipeline"), str(REPO / "llama")]

from data import get_data
from llama_model import LLaMA

import torch
import torch.nn.functional as F


# ------------- 实验配置  -------------------

BASE = {
    "lr": 3e-4,
    "d_model": 384,
    "n_layers": 6,
    "n_heads":    6,
    "n_kv_heads": 2,
    "head_dim":   64,
    "max_steps":  100,
    "seed":       1337,
    "T":          256,
    "B":          32,
    "save_ckpt":  False
}

# 锚点： 预算固定，max_steps 由 B 、 T 反推
TOKEN_BUDGET = 32 * 256 * 100   # 819200

def make_cfg(**overrides):
    
    config = {**BASE, **overrides}
    if "max_steps" not in overrides:
        config["max_steps"] = TOKEN_BUDGET // (config["B"] * config["T"])
    
    return config






@torch.no_grad()
def evaluate(model, val_loader, vocab_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1,vocab_size),
            y.reshape(-1),
            reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens



def run(config):

    torch.manual_seed(config["seed"])
    train_loader, val_loader, meta = get_data(T=config["T"], B=config["B"])
    vocab_size = meta["vocab_size"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    max_steps = config["max_steps"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model = LLaMA(vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim).to(device)
    print(f"[dev] device={device}  cuda_available={torch.cuda.is_available()}model_on={next(model.parameters()).device}")    
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"]
    )
    model.train()

    # --------------- 容器 ---------------------
    history =[]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eval_time = 0.0

    loader_iter = endless(train_loader)
    for step in range(max_steps):
        xb, yb = next(loader_iter)

        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)


        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            yb.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()

        val_loss = None
        if step % (max_steps // 10) == 0:
            torch.cuda.synchronize()
            _e0 = time.perf_counter()

            val_loss = evaluate(model, val_loader, vocab_size, device)

            torch.cuda.synchronize()
            eval_time += time.perf_counter() - _e0
            print(f"step {step:5d}  train {loss.item():.4f}  val {val_loss:.4f}")

        optimizer.step()

        history.append({
            "step" :        step,
            "tokens_seen":  (step+1)*config["B"]*config["T"],
            "train_loss":   loss.item(),
            "val_loss":     val_loss,

        })

    torch.cuda.synchronize()
    _e0 = time.perf_counter()
    
    final_val = evaluate(model, val_loader, vocab_size, device)

    torch.cuda.synchronize()
    eval_time += time.perf_counter() - _e0

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mib = torch.cuda.max_memory_allocated() / 1024 ** 2

    print(f"step {max_steps:5d} train n/a val {final_val:.4f}")
    history.append({
        "step": max_steps,
        "tokens_seen": max_steps * config["B"] * config["T"],
        "train_loss": None,
        "val_loss": final_val,
    })

    # ----------------- 落盘 ---------------------
    out_dir = REPO / "data" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag=make_tag(config)
    out_path = out_dir / f"{tag}.json"

    result = {
        "config":       config,
        "device":       str(device),
        "elapsed_sec":  elapsed,
        "eval_sec":     eval_time,
        "train_sec":    elapsed - eval_time,
        "peak_mem_mib": peak_mib,
        "history":      history,
    }

    with open(out_path,"w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # -------------- 存权重 ---------------------
    ckpt = {
        "config":       config,
        "vocab_size":   vocab_size,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "stoi":         meta["stoi"],
        "itos":         meta["itos"],
        "eos_id":       meta["eos_id"],

    }
    if config["save_ckpt"]:
        ckpt_path = out_dir / f"{tag}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"saved ckpt -> {ckpt_path}")


    
    return history

def endless(loader):
    """ 无限迭代 loader 每个epoch 重新 __iter__ -> RandomSapler 重排。
     不能用 itertools.cycle: 它缓存第一轮的 yield 顺序, 第二轮是复读。 """
    while True:
        yield from loader

RUNS = [
    # 基准 -- 三组实验共用
    make_cfg(save_ckpt=True),

    # 实验 A： lr
    make_cfg(lr=1e-5),
    make_cfg(lr=1e-2),

    # 实验 B: batch
    make_cfg(B=16),
    make_cfg(B=64),

    # 实验 C：context length
    make_cfg(T=64),
    make_cfg(T=128),

    # 线性缩放律对照： lr 正比于 B
    make_cfg(B=64, lr=6e-4),
    make_cfg(max_steps=400, save_ckpt=True),
    make_cfg(max_steps=1600, save_ckpt=True),    
    make_cfg(max_steps=800, save_ckpt =True),
]




if __name__ == "__main__":
    for config in RUNS:
        run(config) 

