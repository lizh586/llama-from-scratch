import sys
from functools import lru_cache
from pathlib import Path
LLM = Path(__file__).resolve().parents[1]
sys.path += [str(LLM/"pipeline"), str(LLM/"sft")]
from data import PAD_ID, T_DEFAULT, B_DEFAULT
from sft_data import render, build_input_ids
import torch
from torch.utils.data import Dataset, DataLoader
B = B_DEFAULT
T = T_DEFAULT


def _pair_to_ids_cached(instruction, input_, chosen, rejected):
    """tokenize 一次就够：2000 步 × 32 条 = 64000 次调用，实际只有 208 个不同 pair。

    pair 是 dict（不可哈希），所以拆成四个字符串做键；语义与原实现逐位一致。
    """
    s_c, n_c = render(instruction, input_, chosen)
    s_r, n_r = render(instruction, input_, rejected)
    ids_c = build_input_ids(s_c)
    ids_r = build_input_ids(s_r)
    return tuple(ids_c), tuple(ids_r)


_pair_to_ids_cached = lru_cache(maxsize=None)(_pair_to_ids_cached)


def pair_to_ids(pair):
    ids_c, ids_r = _pair_to_ids_cached(
        pair["instruction"], pair["input"], pair["chosen"], pair["rejected"])
    return list(ids_c), list(ids_r)

def pad_mask(ids):
    assert len(ids) <= T, "ids 超长 %d > T=%d：pad 变空列表，x 会静默变成 (L,) 而不是 (T,)" % (len(ids), T)
    pad = [PAD_ID] * (T-len(ids))
    x = torch.tensor(pad + ids)
    mask = ((x!= PAD_ID).cumsum(-1) > 0 ).long()
    return x, mask

class RM_Dateset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def  __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        ids_c, ids_r = pair_to_ids(self.pairs[i])
        x_c, m_c = pad_mask(ids_c)
        x_r, m_r = pad_mask(ids_r)
        # 0=chosen，1=rejected
        x = torch.stack([x_c, x_r])
        m = torch.stack([m_c, m_r])
        return x, m
def make_rm_loader(dataset, batch=B, shuffle = True, drop_last = False,num_workers=0):
    return DataLoader(dataset, batch_size=batch,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)