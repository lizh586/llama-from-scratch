import sft_data
import torch
import sys
from pathlib import Path
import json
from torch.utils.data import Dataset ,DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path += [str(REPO / "pipeline")]

from data import VOCAB, EOS_ID, stoi, itos, PAD_ID, T_DEFAULT, B_DEFAULT
""" 
    json 格式
    "instruction": 
    "input":
    "output":
"""
T =  T_DEFAULT  
B = B_DEFAULT

def build_tensor_data(raw_data):
    instruction = raw_data["instruction"]
    input_ = raw_data["input"]
    output = raw_data["output"]
    s, n = sft_data.render(instruction,input_,output)
    ids = sft_data.build_input_ids(s)
    labels = sft_data.build_labels(n, ids)
    x_row = ids + [PAD_ID] * (T - len(ids))
    labels_row = labels + [-100] * (T - len(labels))
    x = torch.tensor(x_row, dtype=torch.int64)
    labels = torch.tensor(labels_row, dtype=torch.int64)


    return x, labels

DATA_PATH = str(REPO / "data" / "sft" / "alpaca_sft.json")


def make_rows(path = DATA_PATH):
    # 把alpaca数据存成 rows
    with open(path, 'r',encoding='utf-8') as f:
        rows = json.load(f)
    return rows

class SFTDataset(Dataset):
    def __init__(self,rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        return build_tensor_data(self.rows[i])
    
def make_sft_loader(dataset, batch=B, shuffle = True, drop_last = True,num_workers=0):
    return DataLoader(dataset, batch_size=batch,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)


if __name__ == "__main__":
    rows = make_rows()
    print(f"rows = {len(rows)}")

    ds = SFTDataset(rows)
    # 演示时关 shuffle：否则 xb[0] 是随机行，下面那个 L0 取自 rows[0]，两者对不上
    loader = make_sft_loader(ds, shuffle=False)
    print(f"len(loader) = {len(loader)} 批   drop_last 丢掉 = {len(rows) - len(loader) * B} 条")

    xb, lb = next(iter(loader))
    print(f"x.shape = {tuple(xb.shape)}   labels.shape = {tuple(lb.shape)}   dtype = {xb.dtype}")
    print(f"labels[0] 里 -100 的个数 = {int((lb[0] == -100).sum())}")

    s, n = sft_data.render(rows[0]["instruction"], rows[0]["input"], rows[0]["output"])
    L0 = len(sft_data.build_input_ids(s))
    print(f"第一行：补长前 L0 = {L0}  ->  补到 T = {T}，补了 {T - L0} 位")


