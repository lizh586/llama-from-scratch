# LLaMA from Scratch

从零手写 LLaMA，并在自写模型上跑通 **Pretrain → SFT** 两阶段训练管线。不用 `transformers`，不用 `trl`。

组件与 PyTorch 参考实现逐项数值比对，误差 < 1e-5（开发期逐组件验证）。仓库内附 3 个可直接运行的测试：
RMSNorm 的数值与梯度、全模型的形状与 `generate` 不变式、attention mask 判别（含一个正控）。

## 目录

```
llama/       组件实现 + 3 个自包含测试
pipeline/    字符级数据管线（切块 / vocab / DataLoader）
pretrain/    预训练 + 采样
sft/         指令微调（prompt 段 mask）+ 不变式校验
```

## 组件

| 组件 | 文件 | 说明 |
|------|------|------|
| RoPE 旋转位置编码 | `llama/rope.py` | 复数推导 → 旋转矩阵 → 频率计算，forward + backward |
| RMSNorm | `llama/rmsnorm.py` | 去中心化归一化，forward + backward |
| SwiGLU 门控 FFN | `llama/swiglu.py` | 门控 + Swish + 线性投影，3 矩阵梯度 finite-diff < 1e-5 |
| GQA 分组查询注意力 | `llama/gqa.py` | KV 头共享 + RoPE + causal mask，MHA 退化 diff=0 |
| Decoder Layer | `llama/llama_block.py` | Pre-Norm + residual，RMSNorm + GQA + SwiGLU 组装 |
| Full LLaMA Model | `llama/llama_model.py` | 端到端前向 + `attn_mask` + `generate`（temperature / top-k / top-p） |

## LLaMA vs GPT-2

| 组件 | GPT-2 | LLaMA |
|------|-------|-------|
| 位置编码 | 绝对位置编码 | **RoPE** 旋转位置编码（复数，支持外推） |
| 归一化 | LayerNorm | **RMSNorm**（去均值，省计算） |
| FFN 激活 | GeLU | **SwiGLU**（门控机制，3 矩阵） |
| 注意力 | MHA | **GQA**（KV 头分组共享，降低推理显存） |

## 训练管线

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 数据 | `pipeline/data.py` | VOCAB 单一来源（printable ASCII + `\n` + `<eos>`，`EOS_ID = PAD_ID`），滑动切块 |
| Pretrain | `pretrain/train.py` | 从零预训练，`run_tag.py` 按超参生成 run tag，落盘 run JSON |
| 采样 | `pretrain/sample.py` | 同一 batch 上多配置采样对比 + `distinct-n` |
| SFT 数据 | `sft/build_alpaca_sft.py` | Alpaca-cleaned → 模板 A/B 渲染 → 按 `T=256` 过滤 |
| SFT | `sft/train_sft.py` | prompt 段 label 置 `-100`，只在 response 段算 loss |
| 校验 | `sft/check_*.py` `sft/measure_*.py` | 见下 |

**SFT 的 mask 是这个仓库最容易错、也最值得看的部分。** `sft/sft_batch.py` 的核心不变式是
`labels[t] == x[t+1]`（非 `-100` 处）—— 补长时 x 补 k 个、labels 补 k+1 个，两者一旦互相借
长度，这条断言必炸。`sft/check_batch.py` 把它落成可失败的断言而不是 print。

**本机跑过的结果**（配置见代码内 `BASE` / `CKPT_TAG`；shakespeare_char 字符级）：

| 阶段 | 步数 | val loss |
|------|------|----------|
| Pretrain `lr0.0003_B32_T256_S800` | 800 | 1.5242 nats |
| SFT `sft_lr1e-05_B32_T256_S1782` | 1782 | 2.9280 → 1.5597（alpaca val，response 口径） |

## 数据与权重

**仓库不含数据集和 checkpoint**（体积原因）。组件测试可以直接跑；训练脚本需要自备下列路径：

```
data/shakespeare_char/          # 预训练语料，字符级
  input.txt  train.bin  val.bin  meta.pkl
data/sft/alpaca_sft.json        # 由 sft/build_alpaca_sft.py 生成
data/runs/                      # 训练产出：*.pt 与 run JSON
```

- `data/shakespeare_char/` —— 用 [nanoGPT](https://github.com/karpathy/nanoGPT) 的
  `data/shakespeare_char/prepare.py` 生成
- `data/sft/alpaca_sft.json` —— `python sft/build_alpaca_sft.py`（需 `datasets`，拉 `tatsu-lab/alpaca`）
- `data/runs/` —— 训练时自动创建

## 运行

```bash
git clone https://github.com/lizh586/llama-from-scratch.git
cd llama-from-scratch
pip install -r requirements.txt

# 组件验证（自包含，不需要数据）
python llama/test_rmsnorm.py
python llama/test_llama_model.py
python llama/test_attn_mask.py

# 训练（先按上节准备好 data/）
python pretrain/train.py
python sft/build_alpaca_sft.py
python sft/train_sft.py
```

## 实现原则

每个组件关掉参考独立手写，forward + backward 逐层推导，与 PyTorch 逐项数值验证。写完以后
Transformer 架构的每个细节不再是黑盒 —— 可以逐层追踪梯度流动。

## 相关项目

- [rl-from-scratch](https://github.com/lizh586/rl-from-scratch) — DQN / PPO / SAC / REINFORCE 从零实现 + DL 核心组件
