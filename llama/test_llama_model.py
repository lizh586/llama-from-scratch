import torch
from llama_model import LLaMA

vocab_size = 1000
d_model = 64
n_layers = 4
n_heads = 8
n_kv_heads = 2
head_dim = 32


gamma = [(torch.ones(d_model), torch.ones(d_model)) for _ in range(n_layers)]
final_gamma = torch.ones(d_model)

model = LLaMA(vocab_size, d_model, n_layers, n_heads, n_kv_heads, head_dim)

B, L = 2, 16
token_ids = torch.randint(0, vocab_size, (B, L))

logits = model.forward(token_ids)

print(f"Input:  {token_ids.shape}")
print(f"Output: {logits.shape}")
assert logits.shape == (B, L, vocab_size), f"Expected {(B, L, vocab_size)}, got {logits.shape}"
print("OK: dimension test passed")

# === Generate tests ===
eos_id = -1  # won't appear in valid token range [0, vocab_size)
prompt = [0, 1, 2, 3, 4]
max_new_tokens = 8
max_len = len(prompt) + max_new_tokens

# Test 1: greedy (temperature=0)
out_greedy = model.generate(prompt.copy(), max_len, eos_id, temperature=0)
assert len(out_greedy) == max_len, f"Greedy: expected len {max_len}, got {len(out_greedy)}"
assert out_greedy[:5] == [0, 1, 2, 3, 4], f"Greedy: prompt modified"
print(f"OK: greedy generate, len={len(out_greedy)}")

# Test 2: sampling with top-k + top-p
out_sample = model.generate(prompt.copy(), max_len, eos_id, temperature=1.0, top_k=10, top_p=0.9)
assert len(out_sample) == max_len, f"Sample: expected len {max_len}, got {len(out_sample)}"
assert out_sample[:5] == [0, 1, 2, 3, 4], f"Sample: prompt modified"
print(f"OK: sampling + top-k/top-p, len={len(out_sample)}")

# Test 3: greedy is deterministic
out_greedy2 = model.generate(prompt.copy(), max_len, eos_id, temperature=0)
assert out_greedy == out_greedy2, "Greedy: not deterministic"
print("OK: greedy deterministic")

# Test 4: EOS termination
eos_real = 5
out_eos = model.generate([0, 1, 2], 100, eos_real, temperature=0)
print(f"OK: EOS test, len={len(out_eos)} (expect early stop if EOS appeared)")

assert len(list(model.parameters()))==38, f"参数不匹配"
assert model.LM_head.weight is model.embedding.weight

