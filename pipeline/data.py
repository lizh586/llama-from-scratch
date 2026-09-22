import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# -------------- 全局默认 ---------------
REPO = Path(__file__).resolve().parents[1]
DATA_PATH = str(REPO / "data" / "shakespeare_char" / "input.txt")
T_DEFAULT = 256
B_DEFAULT = 32

# 固定字符表： printable ASCII(95) + 换行 + 保留 eos 槽位
# 写成 token 列表 防止把 '<eos>' 拆成 5 个 token
VOCAB = [chr(i) for i in range(32, 127)] + ['\n', '<eos>']
EOS_ID = VOCAB.index('<eos>')
PAD_ID = EOS_ID
# 字符↔id 映射，从 VOCAB 派生；全链唯一来源，消费方直接 import，不再各建一份
stoi = {tok : i for i, tok in enumerate(VOCAB)}
itos = {i : tok for i, tok in enumerate(VOCAB)}
vocab_size = len(VOCAB)

# 切块函数
def make_chucks(data, T):
    N = len(data)
    M = N // (T + 1)
    data = data[:M * (T+1)]
    chunks = data.view(M,T+1)
    x = chunks[:,:-1]
    y = chunks[:,1:]
    return x, y

""" def decode(ids):
    return ''.join(itos[int(i)] for i in ids) """

class ShakespeareDateset(Dataset):
    def __init__(self, x, y):
        assert x.shape == y.shape
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# -------------------- 构造流程 ----------------------
def build_data(path=DATA_PATH, T=T_DEFAULT, B=B_DEFAULT):
        
        


    # 1. 读取文件
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. 统计
    total_chars = len(text)     # 总字符数

    preview = text[:200]



    missing = set(text) - set(stoi)
    assert not missing, f"语料含词表外字符： {sorted(missing)}"

    # 编码，把 text 逐字符映射
    text2num = torch.tensor([stoi[ch] for ch in text], dtype=torch.int64)
   
    n = total_chars
    split = int(n * 0.9)

    train_data = text2num[:split]
    val_data = text2num[split:]

    train_x, train_y = make_chucks(train_data, T)
    val_x, val_y = make_chucks(val_data, T)




    train_ds = ShakespeareDateset(train_x, train_y)
    val_ds = ShakespeareDateset(val_x, val_y)

    

    train_loader = DataLoader(
        train_ds, batch_size=B,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds, batch_size=B,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    meta = {
        "text": text,
        "preview": preview,
        "total_chars":total_chars,
        "vocab": VOCAB,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "text2num": text2num,
        "train_data": train_data,
        "val_data": val_data,
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y,
        "train_ds": train_ds, "val_ds": val_ds,
        "T": T, "B": B,
        "eos_id": EOS_ID,
    }
    return train_loader, val_loader, meta

# --------------------- 缓存 ------------------------
_cached = None

def get_data(path=DATA_PATH, T=T_DEFAULT, B=B_DEFAULT):
    global _cached
    if _cached is None or _cached[2]["T"]!=T or _cached[2]["B"]!=B:
        _cached = build_data(path, T, B)
    return _cached


# ------------------- 自检 -----------------------------

if __name__  == '__main__':
    train_loader, val_loader, meta = build_data()
    text        = meta["text"]
    itos        = meta["itos"]
    text2num    = meta["text2num"]
    total_chars = meta["total_chars"]
    vocab       = meta["vocab"]
    vocab_size  = meta["vocab_size"]
    train_data  = meta["train_data"]
    val_data    = meta["val_data"]
    train_x     = meta["train_x"]; train_y = meta["train_y"]
    val_x       = meta["val_x"];   val_y   = meta["val_y"]
    T           = meta["T"];       B       = meta["B"]
    preview = meta["preview"]
    def decode(ids):
        return ''.join(itos[int(i)] for i in ids)

    # 4. 打印
    print(f'总字符数：{total_chars}')
    print(f'字符种类数：{vocab_size}')
    print(f'字符集：{vocab}')
    print(f'前200个字符:')
    print(preview)
    print(f'text2num 字符数： {len(text2num)}')
    print(f'最小值：{min(text2num)}, 最大值：{max(text2num)}')
    n_checks=3; seg_len=16
    starts = random.sample(range(len(text) - seg_len), n_checks)

    for pos in starts:
        ids = text2num[pos : pos+seg_len]
        decoded = ''.join(itos[int(i)] for i in ids)
        original = text[pos : pos + seg_len]
        ok = decoded == original
        assert ok, f'往返验证失败 @ {pos}'

    # assert train_x.shape[0] == len(train_data) // (T+1)
    print(f'[1] 块数 train M = {train_x.shape[0]}, val M = {val_x.shape[0]}')

    for name, t in [('train_x', train_x),('train_y', train_y), ('val_x', val_x),('val_y', val_y)]:
        assert t.shape[1] == T, f'{name} 第二维不是 T'
        assert t.dtype == torch.long, f'{name} dtype 不是 long'
    print(f'[2] 形状 x/y = {tuple(train_x.shape)}, dtype = {train_x.dtype}')

    ok_all = True
    for i in [0, 1, train_x.shape[0] // 2, train_x.shape[0] - 1]:
        same  = torch.equal( train_x[i, 1:], train_y[i, :-1])
        ok_all &= same
        print(f'[3] i={i:>6} y[i][:-1] == x[i][1:] ? {same}')
        assert ok_all
        assert len(train_data) > len(val_data), '90/10 切分反了'


    xb, yb = next(iter(train_loader))
    print(f'[1] xb.shape={tuple(xb.shape)} yb.shape={tuple(yb.shape)} dtype={xb.dtype}')
    assert xb.shape == (B, T) and yb.shape == (B, T)
    assert xb.dtype == torch.long and yb.dtype == torch.long

    x0 = decode(xb[0])
    print(f'[2] x[0] 解码：{x0!r}')

    y0 = decode(yb[0])
    print(f'[3] y[0] 解码：{y0!r}')
    assert y0[:-1] == x0[1:], 'shift 关系破裂！ '
    print(f' -> y[0][:-1] == x[0][1:] 关系成立')

    vb , _ = next(iter(val_loader))
    v0 = decode(vb[0])
    print(f'[4] val_x[0] 解码：{v0!r}')
    assert EOS_ID == 96 and VOCAB[EOS_ID] == '<eos>', f'EOS_ID 契约破裂：{EOS_ID}'