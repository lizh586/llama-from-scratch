import sys
import torch
import torch.nn.functional as F
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path += [str(REPO / "pipeline")]

from data import VOCAB, EOS_ID, stoi, itos

DATA_PATH =str( Path(__file__).resolve().parents[0] / "fixtures.json")

# 读 fixtures.json
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    fixtures = json.load(f)

# render 函数返回 str 和切点 n
def render(instruction, input_, output):
    prompt_str = ""
    if input_ != "":
        prompt_str += "### Instruction:\n"+instruction+"\n\n"+"### Input:\n"+input_+"\n\n"+"### Response:\n"
    elif input_ =="":
        prompt_str +="### Instruction:\n"+instruction+"\n\n"+"### Response:\n"
    n = len(prompt_str)
    s = prompt_str + output
    return s, n

# encode 函数
def encode(text):
    ids = [stoi[ch] for ch in text]
    return ids

# input_ids 函数
def build_input_ids(text):
    return encode(text) + [EOS_ID]
# labels 函数
def build_labels(n, ids):
    return [-100]*(n-1) + ids[n:]



def main():
    s, n = render('Reverse the input string.', 'abc', 'cba') 
    ids = (build_input_ids(s))
    assert n == 74
    assert len(ids) == 78
    assert ids[n:] == [67,66,65,96]
    labels = [-100] * (n-1) + ids[n:]









if __name__ == '__main__':
    main()
