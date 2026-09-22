import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import sft_batch 
import time
import json

REPO = Path(__file__).resolve().parents[1]
sys.path += [str(REPO / "llama"), str(REPO / "pipeline")]

from llama_model import LLaMA
from data  import stoi, itos, vocab_size, EOS_ID

@torch.no_grad()
def evaluate(model, val_loader, vocab_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, labels in val_loader:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1,vocab_size),
            labels.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()
    model.train()
    return total_loss / total_tokens

def endless(loader):
    """ 无限迭代 loader 每个epoch 重新 __iter__ -> RandomSapler 重排。
     不能用 itertools.cycle: 它缓存第一轮的 yield 顺序, 第二轮是复读。 """
    while True:
        yield from loader    












CKPT_TAG = "lr0.0003_B32_T256_S800.pt"
CKPT_PATH = REPO / "data" / "runs" / CKPT_TAG
ckpt = torch.load(CKPT_PATH)

cfg = ckpt["config"]
print(f"ckpt {CKPT_TAG}  config = {cfg}")



torch.manual_seed(cfg["seed"])

rows = sft_batch.make_rows()
n = len(rows)
split = int(n * 0.9)
trian_rows = rows[:split]
val_rows = rows[split:]

train_ds = sft_batch.SFTDataset(trian_rows)
val_ds = sft_batch.SFTDataset(val_rows)
train_loader = sft_batch.make_sft_loader(train_ds,shuffle=True,drop_last=True)
val_loader = sft_batch.make_sft_loader(val_ds,shuffle=False,drop_last=False)

lr = 1e-5
steps_per_epoch = len(train_ds) // cfg["B"] 
max_steps = 3 * steps_per_epoch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"rows = {n}   train = {len(trian_rows)}   val = {len(val_rows)}")
print(f"train_loader = {len(train_loader)} 批   val_loader = {len(val_loader)} 批   max_steps = {max_steps}   device = {device}")


model = LLaMA(
    vocab_size, cfg["d_model"], cfg["n_layers"],
    cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
).to(device)

model.load_state_dict(ckpt["state_dict"])

optimizer = torch.optim.AdamW(
    model.parameters(), lr = lr
)
history = []
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.perf_counter()
eval_time = 0.0


# step-0 基线
print("step 0 基线 evaluate ...")
history.append({
    "step": -1,
    "tokens_seen": 0, 
    "train_loss": None,
    "val_loss": evaluate(model,val_loader,vocab_size,device),
})
print(f"step 0 基线 val_loss = {history[0]['val_loss']:.4f}")

# 训练 loop
loader_iter = endless(train_loader)
for step in range(max_steps):
    x, labels = next(loader_iter)
    x, labels = x.to(device), labels.to(device)

    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1,vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    optimizer.zero_grad()
    loss.backward()

    val_loss = None
    if step % (max_steps // 10) == 0:
        torch.cuda.synchronize()
        _e0 = time.perf_counter()

        val_loss = evaluate(model,val_loader, vocab_size,device)

        torch.cuda.synchronize()
        eval_time += time.perf_counter() - _e0
        print(f"step {step:5d}  train {loss.item():.4f}  val {val_loss:.4f}")
    optimizer.step()

    history.append({
        "step":         step,
        "tokens_seen":  (step+1)*cfg["B"]*cfg["T"],
        "train_loss":   loss.item(),
        "val_loss":     val_loss,
    })
torch.cuda.synchronize()
_e0 = time.perf_counter()

final_val = evaluate(model,val_loader, vocab_size, device)
torch.cuda.synchronize()
eval_time += time.perf_counter() - _e0
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
peak_mib = torch.cuda.max_memory_allocated()/1024 **2

print(f"step {max_steps:5d} train n/a val {final_val:.4f}")
history.append({
    "step":         max_steps,
    "tokens_seen":  max_steps*cfg["B"]*cfg["T"],
    "train_loss":   None,
    "val_loss":     final_val,
})

# --------------- 落盘 --------------------------
out_dir = REPO / "data" / "runs"
out_dir.mkdir(parents=True, exist_ok=True)

tag = f"sft_lr{lr}_B{cfg['B']}_T{cfg['T']}_S{max_steps}"
out_path = out_dir / f"{tag}.json"
config = {
    "lr": lr,
    "d_model": cfg["d_model"],
    "n_layers": cfg["n_layers"],
    "n_heads":    cfg["n_heads"],
    "n_kv_heads": cfg["n_kv_heads"],
    "head_dim":   cfg["head_dim"],
    "max_steps":  max_steps,
    "seed":       cfg["seed"],
    "T":          cfg["T"],
    "B":          cfg["B"],
    "save_ckpt":  True    
}

result = {
    "config":   config,
    "device":   str(device),
    "elapsed_sec":  elapsed,
    "eval_sec":     eval_time,
    "train_sec":    elapsed - eval_time,
    "peak_mem_mib": peak_mib,
    "history":      history,

}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

# -------------- 存权重 -------------------
sft_ckpt = {
    "config":       config,
    "vocab_size":   vocab_size,
    "state_dict":   {k:v.cpu() for k, v in model.state_dict().items()},
    "stoi":         stoi,
    "itos":         itos,
    "eos_id":       EOS_ID,
}
if config["save_ckpt"]:
    ckpt_path = out_dir / f"{tag}.pt"
    torch.save(sft_ckpt, ckpt_path)

