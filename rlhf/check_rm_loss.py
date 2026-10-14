"""rm_loss 数值判据 —— 三条定点 + 形状/口径 + 一次反证（证明这检查能失败）。"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_rm import rm_loss  # noqa: E402


def L_ref(delta: float) -> float:
    """参考值用 float64 + log1p 独立算，不复用被测实现的任何路径。"""
    return math.log1p(math.exp(-delta)) if delta > -700 else -delta


def main():
    ok = True

    # 定点必须用 float32：float64 下 sigmoid 要到 -745 才下溢，-200 测不出稳定形式
    for name, d in [("d=0     ", 0.0), ("d=+10   ", 10.0), ("d=-200  ", -200.0)]:
        want = L_ref(d)
        got = float(rm_loss(torch.tensor([d], dtype=torch.float32)))
        good = math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-6)
        ok &= good
        print(f"{name} got {got!r:24} want {want!r:24} {'OK' if good else 'FAIL'}")

    # 形状 / dtype：0 维标量、float32
    out = rm_loss(torch.tensor([0.0, 10.0], dtype=torch.float32))
    shape_ok = out.dim() == 0 and out.dtype == torch.float32
    ok &= shape_ok
    print(f"shape     got dim={out.dim()} {out.dtype}{'':14} {'OK' if shape_ok else 'FAIL'}")

    # 归约口径：两元素输出 == 两点参考值的均值（改成 sum 这里会 FAIL，属口径不是错）
    want_mean = (L_ref(0.0) + L_ref(10.0)) / 2
    mean_ok = math.isclose(float(out), want_mean, rel_tol=1e-5)
    ok &= mean_ok
    print(f"reduction got {float(out)!r:24} want(mean) {want_mean!r:24} {'OK' if mean_ok else 'FAIL'}")

    print("\n--- 反证：这份检查确实能失败 ---")
    d = torch.tensor([-200.0], dtype=torch.float32)
    naive_a = float(-torch.log(torch.sigmoid(d)))
    naive_b = float(torch.log1p(torch.exp(-d)))
    print(f"-log(sigmoid(Δ))  -> {naive_a}")
    print(f"log1p(exp(-Δ))    -> {naive_b}")
    if math.isfinite(naive_a) and math.isfinite(naive_b):
        print("!! 两种天真写法都没炸 —— 判据点挑错了，这份检查不成立")
        ok = False
    else:
        print("两种天真写法均非有限值 —— 判据有效")

    print("\nRESULT:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
