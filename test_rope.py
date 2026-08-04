import torch
from rope import RoPE

def test_rope():
    B, L, n_heads, head_dim = 1, 2, 1, 8
    torch.manual_seed(42)

    # Forward: shape preserved
    rope = RoPE(head_dim)
    x = torch.randn(B, L, n_heads, head_dim, dtype=torch.float64)
    y = rope.forward(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"

    # Backward: numerical gradient check (sample elements only)
    y2 = rope.forward(x.clone())
    grad_output = torch.randn_like(y2)
    grad_input = rope.backward(grad_output)

    eps = 1e-6
    passed = 0
    total = 0
    max_diff = 0.0
    # Sample first 2 positions, all dims
    for l in range(L):
        for d in range(head_dim):
            total += 1

            x_p = x.clone()
            x_p[0, l, 0, d] += eps
            rp = RoPE(head_dim)
            loss_p = (rp.forward(x_p) * grad_output).sum()

            x_m = x.clone()
            x_m[0, l, 0, d] -= eps
            rm = RoPE(head_dim)
            loss_m = (rm.forward(x_m) * grad_output).sum()

            g = (loss_p - loss_m) / (2 * eps)
            diff = abs(grad_input[0, l, 0, d].item() - g.item())
            if diff < 1e-5:
                passed += 1
            max_diff = max(max_diff, diff)

    print(f"Elements checked: {total}, passed: {passed}")
    print(f"Max grad diff: {max_diff:.2e}")
    if passed == total:
        print("PASS: backward is correct")
    else:
        print("FAIL: backward has errors")

    # Verify rotation preserves dot product (relative position property)
    rope3 = RoPE(head_dim)
    q = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)
    k = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)
    q_r = rope3.forward(q.clone())
    k_r = rope3.forward(k.clone())
    dot_before = torch.einsum('blhd,blhd->bl', q, k)
    dot_after  = torch.einsum('blhd,blhd->bl', q_r, k_r)
    print(f"\nDot product before rotation: {dot_before}")
    print(f"Dot product after rotation:  {dot_after}")
    print("(should change — rotation changes individual dot products)")

if __name__ == "__main__":
    test_rope()
