import torch
from llama_model import LLaMA

torch.manual_seed(0)

vocab_size, d_model, n_layers = 100, 64, 4
n_heads, n_kv_heads, head_dim = 8, 2, 32

model = LLaMA(vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim)
model.eval()

real_len, total_len = 5, 8
n_pad = total_len - real_len

# 同一个真实序列 a，两种 pad 内容（0 和 7），组成 batch=2
a = torch.randint(1, vocab_size, (1, real_len))
pad0 = torch.zeros(1, n_pad, dtype=torch.long)
pad7 = torch.full((1, n_pad), 7, dtype=torch.long)

x_left = torch.cat([torch.cat([pad0, a], dim=1),
                    torch.cat([pad7, a], dim=1)], dim=0)

mask_left = torch.tensor([[0] * n_pad + [1] * real_len] * 2)

with torch.no_grad():
    l_masked = model(x_left, attn_mask=mask_left)
    l_free = model(x_left)
    l_ones = model(x_left, attn_mask=torch.ones_like(mask_left))


def report(tag, got, ref):
    d = (got - ref).abs().max().item()
    print(f"  {tag:40s} max|diff| = {d:.3e}   equal = {torch.equal(got, ref)}")
    return d


print("=== A. backward compat: all-ones mask == None ===")
dA = report("all-ones vs None", l_ones, l_free)

print()
print("=== B. LEFT pad, masked: pad content must not matter ===")
dB = report("(pad=0) vs (pad=7)", l_masked[0, n_pad:], l_masked[1, n_pad:])

print()
print("=== C. LEFT pad, unmasked: this MUST be large ===")
dC = report("(pad=0) vs (pad=7)", l_free[0, n_pad:], l_free[1, n_pad:])

print()
print("=== D. RIGHT pad, masked: causal already blocks it ===")
x_right = torch.cat([torch.cat([a, pad0], dim=1),
                     torch.cat([a, pad7], dim=1)], dim=0)
mask_right = torch.tensor([[1] * real_len + [0] * n_pad] * 2)
with torch.no_grad():
    r_masked = model(x_right, attn_mask=mask_right)
    r_free = model(x_right)
dD1 = report("masked   (pad=0) vs (pad=7)", r_masked[0, :real_len], r_masked[1, :real_len])
dD2 = report("unmasked (pad=0) vs (pad=7)", r_free[0, :real_len], r_free[1, :real_len])

print()
print("=== VERDICT ===")
assert dA == 0.0, f"A 失败：all-ones mask 与 None 不等价（max|diff|={dA:.3e}）"
assert dB == 0.0, f"B 失败：LEFT pad 下 mask 没隔住 pad 内容（max|diff|={dB:.3e}）"
assert dC > 0.01, (
    f"C 正控失败：不 mask 时 pad 内容居然无影响（max|diff|={dC:.3e}）。"
    " 这说明本测试根本区分不出 mask 有没有生效 —— 修它，别改阈值"
)
assert dD1 == 0.0 and dD2 == 0.0, (
    f"D 失败：RIGHT pad 未被 causal 挡住（masked={dD1:.3e} free={dD2:.3e}）"
)
print("A all-ones no-op | B masked pad-independent | C 正控 >0.01 | D right pad causal-blocked")
print("PASS")
