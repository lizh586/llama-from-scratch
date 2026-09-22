import torch
import torch.nn as nn
import torch.nn.functional as F
from rmsnorm import RMSNorm

torch.manual_seed(42)

B, L, H, D = 2, 4, 3, 8
x = torch.randn(B, L, H, D, dtype=torch.float64)

norm = RMSNorm(D).double()
# gamma 用非平凡值：gamma=1 会盖住「gamma 根本没被用上」这类错
with torch.no_grad():
    norm.gamma.copy_(torch.randn(D, dtype=torch.float64))


def reference(x, gamma, eps):
    """测试侧独立实现 —— 不调用被测代码，否则是循环论证"""
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * gamma


# ---- 1. forward 数值对拍 ----
y = norm(x.clone())
assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

diff_fwd = (y - reference(x, norm.gamma.detach(), norm.eps)).abs().max().item()
print(f"Forward max diff vs manual: {diff_fwd:.2e}")
assert diff_fwd < 1e-10, f"forward 与手算 RMSNorm 不一致: {diff_fwd:.2e}"

# ---- 2. 与 PyTorch 官方实现对拍 ----
y_torch = F.rms_norm(x, (D,), norm.gamma.detach(), norm.eps)
diff_torch = (y - y_torch).abs().max().item()
print(f"Forward max diff vs torch.nn: {diff_torch:.2e}")
assert diff_torch < 1e-10, f"与 F.rms_norm 不一致: {diff_torch:.2e}"

# ---- 3. autograd 是不是真的连上了 ----
x_in = x.clone().requires_grad_(True)
grad_output = torch.randn(B, L, H, D, dtype=torch.float64)

y = norm(x_in)
(y * grad_output).sum().backward()

assert x_in.grad is not None, "梯度没到输入 x"
assert norm.gamma.grad is not None, "梯度没到 gamma —— gamma 不在计算图里"
print("OK: autograd reaches both x and gamma")

# ---- 4. 有限差分校验梯度（替代原先手写 backward 的对拍）----
fd = 1e-6
gg = norm.gamma.detach()


def numeric_grad_gamma(i):
    gp = gg.clone(); gp[i] += fd
    gm = gg.clone(); gm[i] -= fd
    lp = (reference(x, gp, norm.eps) * grad_output).sum()
    lm = (reference(x, gm, norm.eps) * grad_output).sum()
    return ((lp - lm) / (2 * fd)).item()


worst_g = 0.0
for i in range(D):
    worst_g = max(worst_g, abs(numeric_grad_gamma(i) - norm.gamma.grad[i].item()))

worst_x = 0.0
for l in range(L):
    for d in range(D):
        xp = x.clone(); xp[0, l, 0, d] += fd
        xm = x.clone(); xm[0, l, 0, d] -= fd
        lp = (reference(xp, gg, norm.eps) * grad_output).sum()
        lm = (reference(xm, gg, norm.eps) * grad_output).sum()
        worst_x = max(worst_x, abs(((lp - lm) / (2 * fd)).item() - x_in.grad[0, l, 0, d].item()))

print(f"Grad finite-diff max diff:  gamma={worst_g:.2e}  x={worst_x:.2e}")
assert worst_g < 1e-6, f"gamma 梯度不对: {worst_g:.2e}"
assert worst_x < 1e-6, f"x 梯度不对: {worst_x:.2e}"
print("PASS")
