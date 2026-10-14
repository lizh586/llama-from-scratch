import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from rm_batch import RM_Dateset, make_rm_loader
from rm import RM

from paths import ROOT  # noqa: E402
LLM = Path(__file__).resolve().parents[0]
sys.path += [str(LLM/"pipeline")]
from data import stoi, itos, EOS_ID, vocab_size  # noqa: E402

DATAPATH = str(ROOT/"data"/"rlhf"/"toy_pairs.json")
SFT_CKPT = ROOT/"data"/"runs"/"sft_lr1e-05_B32_T256_S1782.pt"

def rm_loss(delta):
    return F.softplus(-delta).mean()

def compute_delta(model, x, m):
   
    r_c = model(x[:,0], m[:,0])
    r_r = model(x[:,1], m[:,1])
    delta = r_c - r_r 
    return delta   

def endless(loader):
    while True:
        yield from loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4, help="scalar_head 的 lr")
    ap.add_argument("--lr-backbone", type=float, default=None,
                    help="backbone 的 lr；不填 = 与 --lr 相同（分离 lr 是 10/1 深夜实验引入的）")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default=None, help="默认 cuda（可用则用）")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    lr_bb = args.lr if args.lr_backbone is None else args.lr_backbone
    dev = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)  # make_rm_loader 默认 shuffle=True，种住才能复现同一批
    with open(DATAPATH, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    train = pairs["train"]
    caliber = pairs["caliber"]
    ds = RM_Dateset(train)
    loader = make_rm_loader(ds)

    model = RM(str(SFT_CKPT)).to(dev)
    print(f"device = {dev}  steps = {args.steps}  lr_head = {args.lr:g}  lr_backbone = {lr_bb:g}")

    # 分离 lr：不能用 model.parameters()（会把两组参数重复交给优化器）
    head_params = list(model.scalar_head.parameters())
    bb_params = list(model.model.parameters())
    opt = torch.optim.Adam([
        {"params": bb_params, "lr": lr_bb},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    bb0 = {k: v.detach().clone() for k, v in model.model.state_dict().items()}

    loader_iterator = endless(loader)
    for step in range(args.steps):
        x, m = next(loader_iterator)
        x, m = x.to(dev), m.to(dev)
        delta = compute_delta(model, x, m)
        loss = rm_loss(delta)
        opt.zero_grad()
        loss.backward()
        gmax = model.scalar_head.weight.grad.abs().max().item()
        gmax_bb = max(p.grad.abs().max().item() for p in bb_params if p.grad is not None)
        opt.step()
        if step < 5 or step % 50 == 49:
            print(f"{step:5d}  loss {loss.item():.6e}  "
                f"|d|mean {delta.abs().mean().item():8.4f}  "
                f"|d|max {delta.abs().max().item():8.4f}  "
                f"gmax_head {gmax:.3e}  gmax_bb {gmax_bb:.3e}")
    delta_before = compute_delta(model, x, m)

    # 表示到底动没动：backbone 相对初始权重的漂移
    bb1 = model.model.state_dict()
    drifts = [(bb1[k] - bb0[k]).abs().mean().item() for k in bb0]
    scale = sum(v.abs().mean().item() for v in bb0.values()) / len(bb0)
    drift = sum(drifts) / len(drifts)
    print(f"backbone drift: mean|d| = {drift:.3e}   权重平均量级 {scale:.3e}   相对 {drift/scale:.2%}")

    rm_ckpt = {
        "config":model.config,
        "vocab_size":vocab_size,
        "state_dict":{k: v.cpu() for k, v in model.state_dict().items()},
        "stoi":stoi,
        "itos":itos,
        "eos_id":EOS_ID,
        "caliber":caliber
    }
    tag = args.tag or (f"rm_v3_S{args.steps}_lrh{args.lr:g}_lrb{lr_bb:g}"
                       f"{'_wd' + format(args.weight_decay, 'g') if args.weight_decay else ''}")
    path = ROOT / "data" / "runs"/f"{tag}.pt"
    torch.save(rm_ckpt, path)
    print("saved", path)
    model2 = RM(str(path)).to(dev)   # 同设备比较：换设备会换 kernel，逐位不可比
    delta_after = compute_delta(model2, x, m)
    print("round-trip identical:", torch.equal(delta_before, delta_after))
    





if __name__ == "__main__":
    main()
