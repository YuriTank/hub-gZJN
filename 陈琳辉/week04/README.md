# Transformer编码器层实现 - Week04作业

本项目是从零实现了一个完整的Transformer编码器层，包含详细的中文注释和清晰的代码结构，适合学习和理解Transformer的核心原理。

## 📁 文件结构

```
.
├── transformer_layer.py  # 核心代码文件
└── README.md             # 本文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+

### 安装依赖

```bash
pip install torch
```

### 运行测试

```bash
python transformer_layer.py
```

## 🏗️ 核心组件

### 1. MultiHeadAttention

多头自注意力机制，支持多头并行计算注意力权重，从不同表示子空间捕捉信息。

```python
attention = MultiHeadAttention(d_model=512, num_heads=8)
output, weights = attention(query, key, value)
```

### 2. FeedForwardNetwork

前馈神经网络，对每个位置独立应用的两层全连接网络。

```python
ffn = FeedForwardNetwork(d_model=512, d_ff=2048)
output = ffn(x)
```

### 3. TransformerEncoderLayer

单个Transformer编码器层，包含：
- 多头自注意力
- 残差连接 + 层归一化
- 前馈神经网络
- 残差连接 + 层归一化

```python
layer = TransformerEncoderLayer(d_model=512, num_heads=8, d_ff=2048)
output = layer(x)
```

### 4. TransformerEncoder

完整的Transformer编码器，由多个编码器层堆叠而成。

```python
encoder = TransformerEncoder(num_layers=6, d_model=512, num_heads=8, d_ff=2048)
output = encoder(x)
```

## 📊 代码使用示例

```python
import torch
from transformer_layer import TransformerEncoder

# 参数配置
batch_size = 2
seq_len = 10
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

# 创建编码器
encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)

# 创建随机输入
x = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output = encoder(x)

print(f"输出形状: {output.shape}")  # [2, 10, 512]
```

## 📐 架构图

```
输入
  │
  ├──────────────────────────────┬─────────────────────────────────────┐
  │                              │                                     │
  ▼                              ▼                                     │
[Multi-Head          ───► Add & LayerNorm ─────────────────────────────┤
 Self-Attention]               │                                      │
  │                             │                                      │
  │    (残差连接)                │                                      │
  │                             │                                      │
  │                             ▼                                      │
  │                    [Feed-Forward]                                   │
  │                             │                                      │
  │                             ▼                                      │
  └────────────────────────► Add & LayerNorm ──► 输出
                             (残差连接)
```

## 🎯 核心参数说明

| 参数 | 含义 | 常用值 |
|------|------|--------|
| `d_model` | 模型的维度，决定了表示能力的容量 | 512, 768 |
| `num_heads` | 注意力头的数量 | 8, 12, 16 |
| `d_ff` | 前馈网络中间层维度 | 2048, 3072 |
| `num_layers` | 编码器层数 | 6, 12 |
| `dropout` | 正则化比率 | 0.1, 0.2 |

## 🔑 设计特点

- **清晰的注释**：所有代码包含详细的中文注释
- **模块化设计**：每个组件独立实现，便于理解和复用
- **残差连接**：缓解深层网络的梯度消失问题
- **层归一化**：稳定训练过程
- **完整的测试**：包含完整的测试代码，可以直接运行

## 📚 理论基础

### 注意力机制公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### 前馈网络公式

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```


