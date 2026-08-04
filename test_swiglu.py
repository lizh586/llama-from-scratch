import torch
import torch.nn.functional as F
from swiglu import SwiGLU


def numerical_grad_check():
    """finite difference 验证手写 backward"""
    torch.manual_seed(42)
    d_model = 6
    batch = 3

    swiglu = SwiGLU(d_model)
    swiglu.W_gate = torch.randn(d_model, swiglu.W_gate.shape[1], dtype=torch.float64) * 0.5
    swiglu.W_up   = torch.randn(d_model, swiglu.W_up.shape[1], dtype=torch.float64) * 0.5
    swiglu.W_down = torch.randn(swiglu.W_down.shape[0], d_model, dtype=torch.float64) * 0.5

    x = torch.randn(batch, d_model, dtype=torch.float64)

    # 用线性 loss: L = sum(d_upstream * output)
    # 这样 dL/d(output) = d_upstream，精确对应 backward(d_upstream)
    d_upstream = torch.randn(batch, d_model, dtype=torch.float64)

    # analytical gradients
    output = swiglu.forward(x)
    dx_analytical = swiglu.backward(d_upstream)
    dW_gate_analytical = swiglu.dW_gate
    dW_up_analytical   = swiglu.dW_up
    dW_down_analytical = swiglu.dW_down

    eps = 1e-5
    tol = 1e-5
    params = [
        ("W_gate", swiglu.W_gate, dW_gate_analytical),
        ("W_up",   swiglu.W_up,   dW_up_analytical),
        ("W_down", swiglu.W_down, dW_down_analytical),
    ]

    all_ok = True
    for name, W, dW_ana in params:
        grad_numerical = torch.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] += eps
                out_p = swiglu.forward(x)
                loss_p = (d_upstream * out_p).sum()
                W[i, j] -= 2 * eps
                out_m = swiglu.forward(x)
                loss_m = (d_upstream * out_m).sum()
                W[i, j] += eps
                grad_numerical[i, j] = (loss_p - loss_m) / (2 * eps)

        diff = (grad_numerical - dW_ana).abs().max().item()
        status = "OK" if diff < tol else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  d{name}: max diff = {diff:.2e}  [{status}]")

    # dx check
    x_copy = x.clone()
    dx_numerical = torch.zeros_like(x)
    for i in range(batch):
        for j in range(d_model):
            x_copy[i, j] = x[i, j] + eps
            out_p = swiglu.forward(x_copy)
            loss_p = (d_upstream * out_p).sum()
            x_copy[i, j] = x[i, j] - eps
            out_m = swiglu.forward(x_copy)
            loss_m = (d_upstream * out_m).sum()
            x_copy[i, j] = x[i, j]
            dx_numerical[i, j] = (loss_p - loss_m) / (2 * eps)

    dx_diff = (dx_numerical - dx_analytical).abs().max().item()
    dx_status = "OK" if dx_diff < tol else "FAIL"
    if dx_status == "FAIL":
        all_ok = False
    print(f"  dx:     max diff = {dx_diff:.2e}  [{dx_status}]")

    if all_ok:
        print("\n  All gradients pass (< 1e-5)")
    else:
        print("\n  Some gradients FAIL — check backward logic")
    return all_ok


def param_count_comparison():
    """标准 FFN (GELU) vs SwiGLU — 参数量对比"""
    print("\n--- 参数对比 ---")
    for d in [256, 512, 1024, 2048, 4096]:
        d_ff_std = 4 * d
        d_ff_gated = int(8 * d / 3)
        params_std = 2 * d * d_ff_std
        params_swiglu = 3 * d * d_ff_gated
        print(f"  d={d:4d} | std FFN d_ff={d_ff_std:5d} ({params_std/1e6:.2f}M params)"
              f" | SwiGLU d_ff={d_ff_gated:5d} ({params_swiglu/1e6:.2f}M params)"
              f" | ratio={params_swiglu/params_std:.3f}")


def param_detail():
    print("\n--- 具体例子: d_model=6 ---")
    d = 6
    d_ff_std = 4 * d
    d_ff_gated = int(8 * d / 3)
    print(f"  标准 FFN:  W1={d}x{d_ff_std}, W2={d_ff_std}x{d}  → {2*d*d_ff_std} params")
    print(f"  SwiGLU:    W_gate={d}x{d_ff_gated}, W_up={d}x{d_ff_gated}, W_down={d_ff_gated}x{d}  → {3*d*d_ff_gated} params")
    print(f"  ratio = {3*d*d_ff_gated} / {2*d*d_ff_std} = {3*d*d_ff_gated / (2*d*d_ff_std):.3f}")


if __name__ == "__main__":
    print("=== SwiGLU Gradient Check ===")
    ok = numerical_grad_check()
    param_count_comparison()
    param_detail()
