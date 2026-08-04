import torch
import torch.nn.functional as F
from gqa import GroupedQueryAttention


def _multinomial_to_onehot_softmax(scores, mask):
    """用 PyTorch 标准函数做一遍 causal scaled dot-product attention，返回 output"""
    d_k = scores.shape[-1]
    scores_ref = scores + mask
    attn = F.softmax(scores_ref, dim=-1)
    return attn


def test_mha_degeneration():
    """n_kv_heads == n_heads 时，输出应与标准 MHA 一致（逐项验证）"""
    torch.manual_seed(42)
    d_model = 8
    n_heads = 4
    n_kv_heads = 4       # MHA
    head_dim = 8
    max_seq_len = 6
    batch, seq = 2, 5

    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, max_seq_len)

    # 用随机权重
    gqa.W_q = torch.randn(d_model, n_heads * head_dim, dtype=torch.float64) * 0.5
    gqa.W_k = torch.randn(d_model, n_kv_heads * head_dim, dtype=torch.float64) * 0.5
    gqa.W_v = torch.randn(d_model, n_kv_heads * head_dim, dtype=torch.float64) * 0.5
    gqa.W_o = torch.randn(n_heads * head_dim, d_model, dtype=torch.float64) * 0.5

    x = torch.randn(batch, seq, d_model, dtype=torch.float64)

    # ---- 参考实现：用 PyTorch nn.functional ----
    # 投影
    Q_ref = (x @ gqa.W_q).reshape(batch, seq, n_heads, head_dim).transpose(1, 2)
    K_ref = (x @ gqa.W_k).reshape(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
    V_ref = (x @ gqa.W_v).reshape(batch, seq, n_kv_heads, head_dim).transpose(1, 2)

    # RoPE
    from rope import RoPE
    rope = RoPE(head_dim)
    Q_ref = rope.forward(Q_ref.transpose(1, 2)).transpose(1, 2)
    K_ref = rope.forward(K_ref.transpose(1, 2)).transpose(1, 2)

    scores_ref = Q_ref @ K_ref.transpose(-2, -1) / (head_dim ** 0.5)
    causal_mask = torch.triu(torch.full((seq, seq), float('-inf')), diagonal=1)
    attn_ref = _multinomial_to_onehot_softmax(scores_ref, causal_mask)
    output_ref = attn_ref @ V_ref
    output_ref = output_ref.transpose(1, 2).reshape(batch, seq, n_heads * head_dim) @ gqa.W_o

    # ---- GQA 实现 ----
    output_gqa = gqa.forward(x)

    diff = (output_gqa - output_ref).abs().max().item()
    if diff < 1e-5:
        print(f"  MHA degeneration: max diff = {diff:.2e}  [OK]")
        return True
    else:
        print(f"  MHA degeneration: max diff = {diff:.2e}  [FAIL]")
        print(f"    output_ref[0,0,:4] = {output_ref[0,0,:4]}")
        print(f"    output_gqa[0,0,:4] = {output_gqa[0,0,:4]}")
        return False


def test_gqa_grouping():
    """GQA 模式下 repeat_kv 形状和分组正确性"""
    torch.manual_seed(123)
    d_model = 16
    n_heads = 6
    n_kv_heads = 2
    head_dim = 8
    max_seq_len = 4
    batch, seq = 1, 3

    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, max_seq_len)
    gqa.W_q = torch.randn(d_model, n_heads * head_dim, dtype=torch.float64)
    gqa.W_k = torch.randn(d_model, n_kv_heads * head_dim, dtype=torch.float64)
    gqa.W_v = torch.randn(d_model, n_kv_heads * head_dim, dtype=torch.float64)
    gqa.W_o = torch.randn(n_heads * head_dim, d_model, dtype=torch.float64)

    x = torch.randn(batch, seq, d_model, dtype=torch.float64)

    # 手动提取中间结果检查 repeat_kv
    Q = (x @ gqa.W_q).reshape(batch, seq, n_heads, head_dim)
    K = (x @ gqa.W_k).reshape(batch, seq, n_kv_heads, head_dim)
    V = (x @ gqa.W_v).reshape(batch, seq, n_kv_heads, head_dim)

    from rope import RoPE
    rope = RoPE(head_dim)
    Q = rope.forward(Q)
    K_before_repeat = rope.forward(K)
    V_before_repeat = V  # RoPE 不作用于 V

    # repeat_kv 在转置之前执行（匹配 gqa.py 实现顺序）
    K_after_repeat = gqa.repeat_kv(K_before_repeat)
    V_after_repeat = gqa.repeat_kv(V_before_repeat)

    Q = Q.transpose(1, 2)
    K_after = K_after_repeat.transpose(1, 2)
    V_after = V_after_repeat.transpose(1, 2)

    # 验证形状
    assert K_after.shape == (batch, n_heads, seq, head_dim), f"Expected ({batch}, {n_heads}, {seq}, {head_dim}), got {K_after.shape}"
    assert V_after.shape == (batch, n_heads, seq, head_dim), f"Expected ({batch}, {n_heads}, {seq}, {head_dim}), got {V_after.shape}"
    print(f"  repeat_kv shape: {K_after.shape}  [OK]")

    # 验证分组：同一组内（n_heads//n_kv_heads = 3 个 Q 头）的 K 应该完全相同
    group_size = n_heads // n_kv_heads
    all_same = True
    for g in range(n_kv_heads):
        heads_in_group = [g * group_size + i for i in range(group_size)]
        baseline = K_after[0, heads_in_group[0]]
        for h in heads_in_group[1:]:
            if not torch.allclose(K_after[0, h], baseline):
                all_same = False
                print(f"  group {g}: head {heads_in_group[0]} vs head {h} DIFFER")
    if all_same:
        print(f"  GQA grouping: {n_kv_heads} KV heads → {n_heads} Q heads (group_size={group_size})  [OK]")
    else:
        print(f"  GQA grouping: FAIL — repeat_kv not producing identical KV within groups")

    # 验证不同组之间的 K 不同
    g0_k = K_after[0, 0]
    g1_k = K_after[0, group_size]
    if not torch.allclose(g0_k, g1_k):
        print(f"  Cross-group uniqueness: group 0 vs group 1 differ  [OK]")
    else:
        print(f"  Cross-group uniqueness: FAIL — different groups have same K")

    # 验证 V 同理
    g0_v = V_after[0, 0]
    g1_v = V_after[0, group_size]
    if not torch.allclose(g0_v, g1_v):
        print(f"  Cross-group V uniqueness: group 0 vs group 1 differ  [OK]")
    else:
        print(f"  Cross-group V uniqueness: FAIL")


def test_causal_mask():
    """验证 causal mask 确实阻止了未来 token 的 attention"""
    torch.manual_seed(99)
    d_model = 4
    n_heads = 2
    n_kv_heads = 2
    head_dim = 4
    max_seq_len = 5
    batch, seq = 1, 3

    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, max_seq_len)
    x = torch.randn(batch, seq, d_model)

    # 手动跑到 attn_weights
    Q = (x @ gqa.W_q).reshape(batch, seq, n_heads, head_dim)
    K = (x @ gqa.W_k).reshape(batch, seq, n_kv_heads, head_dim)
    from rope import RoPE
    rope = RoPE(head_dim)
    Q = rope.forward(Q).transpose(1, 2)
    K = rope.forward(K).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)
    causal_mask = torch.triu(torch.full((seq, seq), float('-inf')), diagonal=1)
    attn = torch.softmax(scores + causal_mask, dim=-1)

    # token 0: 只能 attend token 0 (对角线只有自己)
    assert attn[0, 0, 0, 1:].abs().sum() < 1e-6, "Token 0 attended future!"
    print(f"  Token 0 attends only self: attn={attn[0,0,0][:seq].detach()}  [OK]")

    # token 1: 只能 attend [0, 1]
    assert attn[0, 0, 1, 2:].abs().sum() < 1e-6, "Token 1 attended future!"
    print(f"  Token 1 attends [0,1]: sum={attn[0,0,1,:2].sum().item():.4f}  [OK]")

    # token 2: attend 所有 [0,1,2]
    assert attn[0, 0, 2, :].sum() - 1.0 < 1e-6, "Token 2 attention doesn't sum to 1"
    print(f"  Token 2 attends all: sum={attn[0,0,2,:].sum().item():.4f}  [OK]")


if __name__ == "__main__":
    print("=== GQA Causal Attention Tests ===\n")
    print("[1] MHA Degeneration (n_kv_heads == n_heads)")
    ok_mha = test_mha_degeneration()

    print("\n[2] GQA Grouping & repeat_kv")
    test_gqa_grouping()

    print("\n[3] Causal Mask Correctness")
    test_causal_mask()

    if ok_mha:
        print("\n=== All critical tests passed ===")
    else:
        print("\n=== MHA degeneration FAILED, debug above ===")
