# LLaMA from Scratch

从零手写 LLaMA 大语言模型核心架构，6 组件独立实现 + 手写反向传播，与 PyTorch 参考实现误差 < 1e-5。

## 组件

| 组件 | 文件 | 说明 |
|------|------|------|
| RoPE 旋转位置编码 | `rope.py` | 复数推导 → 旋转矩阵 → 频率计算，forward + backward |
| RMSNorm | `rmsnorm.py` | 去中心化归一化，forward + backward |
| SwiGLU 门控 FFN | `swiglu.py` | 门控 + Swish + 线性投影，3 矩阵梯度 finite-diff < 1e-5 |
| GQA 分组查询注意力 | `gqa.py` | KV 头共享 + RoPE + causal mask，MHA 退化 diff=0 |
| Decoder Layer | `llama_block.py` | Pre-Norm + residual，RMSNorm + GQA + SwiGLU 组装 |
| Full LLaMA Model | `llama_model.py` | 端到端前向 + generate（top-k / top-p / temperature） |

## LLaMA vs GPT-2

| 组件 | GPT-2 | LLaMA |
|------|-------|-------|
| 位置编码 | 绝对位置编码 | **RoPE** 旋转位置编码（复数，支持外推） |
| 归一化 | LayerNorm | **RMSNorm**（去均值，省计算） |
| FFN 激活 | GeLU | **SwiGLU**（门控机制，3 矩阵） |
| 注意力 | MHA | **GQA**（KV 头分组共享，降低 KV-cache 显存） |

## 运行

```bash
git clone https://github.com/lizh586/llama-from-scratch.git
cd llama-from-scratch
pip install torch numpy

# 逐组件验证
python test_rope.py
python test_rmsnorm.py
python test_swiglu.py
python test_gqa.py
python test_llama_block.py
python test_llama_model.py
```

## 实现原则

每个组件关掉参考独立手写，forward + backward 逐层推导，与 PyTorch 逐项数值验证。写完以后 Transformer 架构的每个细节不再是黑盒——可以逐层追踪梯度流动。

## 相关项目

- [rl-from-scratch](https://github.com/lizh586/rl-from-scratch) — DQN / PPO / SAC 从零实现 + DL 核心组件
