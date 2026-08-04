import torch
from rmsnorm import RMSNorm

torch.manual_seed(42)

B, L, H, D = 2, 4, 3, 8
x = torch.randn(B, L, H, D, dtype=torch.float64)
gamma = torch.randn(D, dtype=torch.float64)

norm = RMSNorm()

# Forward
y = norm.forward(x.clone(), gamma)
print(f"Forward output shape: {y.shape}")
assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

# Verify forward against manual RMSNorm computation
rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True))
y_expected = x / rms * gamma
diff_fwd = (y - y_expected).abs().max().item()
print(f"Forward max diff vs manual: {diff_fwd:.2e}")

# Backward: finite diff check
grad_output = torch.randn_like(y)
grad_input = norm.backward(grad_output)

eps = 1e-6
max_diff = 0.0
failed = 0
total = 0

for l in range(L):
    for d in range(D):
        total += 1

        x_p = x.clone()
        x_p[0, l, 0, d] += eps
        n_p = RMSNorm()
        loss_p = (n_p.forward(x_p, gamma) * grad_output).sum()

        x_m = x.clone()
        x_m[0, l, 0, d] -= eps
        n_m = RMSNorm()
        loss_m = (n_m.forward(x_m, gamma) * grad_output).sum()

        g_numeric = (loss_p - loss_m) / (2 * eps)
        diff = abs(g_numeric.item() - grad_input[0, l, 0, d].item())
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            failed += 1

print(f"Backward: {total - failed}/{total} passed, max diff: {max_diff:.2e}")
if failed == 0:
    print("PASS")
else:
    print(f"FAIL: {failed} elements exceed tolerance")

# Compare with torch.nn.functional (PyTorch >= 2.0 has RMSNorm)
try:
    import torch.nn.functional as F
    if hasattr(F, 'rms_norm'):
        y_torch = F.rms_norm(x, (D,), gamma)
        diff_torch = (y - y_torch).abs().max().item()
        print(f"Forward max diff vs torch.nn: {diff_torch:.2e}")
    else:
        print("torch.nn.functional.rms_norm not available (requires PyTorch >= 2.0)")
except Exception as e:
    print(f"Could not compare with torch: {e}")
