# Transformer Encoder 手动实现指南

> 本 README 记录 Transformer Encoder Layer 的整体计算流程与实现细节，供实现时查阅。

---

## 一、整体架构（Encoder-Only）

```
Input Token IDs
    ↓
[Token Embedding] + [Position Embedding]   ← 输入层（非必须，看你需求）
    ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│  + Add & Norm（残差 + LayerNorm）    │
├─────────────────────────────────────┤
│  Feed-Forward Network (FFN)         │
│  + Add & Norm（残差 + LayerNorm）    │
└─────────────────────────────────────┘
    ↓ 可以堆叠 N 个这样的 Block
Output (与输入同 shape)
```

---

## 二、输入层（Embedding）

### 2.1 Token Embedding
- 将离散的 token ID `(batch, seq_len)` 映射到连续向量 `(batch, seq_len, d_model)`
- 使用 `nn.Embedding(vocab_size, d_model)`

### 2.2 Position Embedding
- 原始论文使用**正弦/余弦函数**编码位置信息（与 token embedding 相加）
- BERT 使用**可学习的 Position Embedding**（也是 `nn.Embedding`，但输入是位置索引 0,1,2...）
- **注意**：如果你要复现原始论文，应该用正弦编码；BERT 是可学习编码

### 2.3 输入注意事项
- `nn.Embedding` 的输入必须是 **LongTensor**（token ID 是整数）
- 如果你测试时直接 `torch.randn(batch, seq, d_model)`，那是已经嵌入后的浮点向量，不需要再过 Embedding 层

---

## 三、多头自注意力（Multi-Head Self-Attention）

### 3.1 核心思想
将 `d_model` 维空间分成 `nhead` 个独立的子空间，每个头在 `d_k = d_model / nhead` 维上并行计算注意力。

### 3.2 计算流程

```
输入 x: (batch, seq_len, d_model)

Step 1: 线性投影得到 Q, K, V
    Q = x @ W_q    → (batch, seq_len, d_model)
    K = x @ W_k    → (batch, seq_len, d_model)
    V = x @ W_v    → (batch, seq_len, d_model)

> **为什么用 `nn.Linear` 而不是手动创建 `W` 矩阵？**
> 
> 原始论文公式就是 `Q = XW^Q + b^Q`，偏置项本来就存在。PyTorch 官方 `nn.MultiheadAttention` 默认也是 `bias=True`。
> 
> 手动写当然也可以：
> ```python
> self.W_q = nn.Parameter(torch.randn(d_model, d_model))
> self.b_q = nn.Parameter(torch.zeros(d_model))
> Q = x @ self.W_q + self.b_q
> ```
> 这和 `nn.Linear(d_model, d_model)` **数学完全等价**，但 Linear 帮你做了合理的初始化、参数注册和高效前向。如果你确实不想要偏置，可以写 `nn.Linear(..., bias=False)`，但通常保留。

Step 2: Split into multiple heads（分头）
    把最后一个维度 d_model 拆成 (nhead, d_k)
    Q → (batch, nhead, seq_len, d_k)
    K → (batch, nhead, seq_len, d_k)
    V → (batch, nhead, seq_len, d_k)
    
    PyTorch 写法：view + transpose
    q = Q.view(batch, seq_len, nhead, d_k).transpose(1, 2)

Step 3: 计算 Scaled Dot-Product Attention（单头内）
    scores = Q @ K^T / sqrt(d_k)
           → (batch, nhead, seq_len, seq_len)
    
    如果有 attention_mask（如 pad_mask），在 softmax 前给 scores 加 mask
    （通常把 pad 位置设为 -inf，softmax 后变为 0）
    
    attn_weights = softmax(scores, dim=-1)
    attn_weights = dropout(attn_weights)
    
    output = attn_weights @ V
           → (batch, nhead, seq_len, d_k)

Step 4: Concat heads（合并头）
    把 nhead 和 d_k 合并回 d_model
    output → (batch, seq_len, nhead, d_k)
    output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

Step 5: 输出线性投影
    output = output @ W_o    → (batch, seq_len, d_model)
```

### 3.3 关于 Mask
- **Padding Mask**：忽略 `<pad>` token，避免填充位置参与注意力计算
- **Causal Mask（Look-ahead Mask）**：Decoder 中使用，防止看到未来信息；Encoder 不需要

---

### 3.4 PyTorch 张量操作详解：`@`、`view` 与 `transpose`

#### `@` 是矩阵乘法

在 PyTorch 中，`@` 是 `torch.matmul` 的语法糖，支持**自动广播 batch 维度**：

```python
# 2D: 普通矩阵乘法
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B  # → (3, 5)

# 3D/4D: 对最后两个维度做矩阵乘法，前面维度 broadcast
A = torch.randn(2, 3, 4)   # 2个 (3×4) 矩阵
B = torch.randn(2, 4, 5)   # 2个 (4×5) 矩阵
C = A @ B  # → (2, 3, 5)  # 逐矩阵相乘

# 4D 多头场景
A = torch.randn(2, 8, 10, 64)  # (batch, nhead, seq, d_k)
B = torch.randn(2, 8, 64, 10)  # (batch, nhead, d_k, seq)
C = A @ B  # → (2, 8, 10, 10)
```

> 多头拆分后得到 `(batch, nhead, seq, d_k)`，计算 `Q @ K.transpose(-2, -1)` 会得到 `(batch, nhead, seq, seq)`，因为最后两维是 `(seq, d_k) @ (d_k, seq)`。

#### `view` vs `transpose`

| | `transpose(dim0, dim1)` | `view(*shape)` |
|---|---|---|
| **作用** | 交换两个维度的**顺序** | 改变 tensor 的**形状**（元素总数不变） |
| **内存** | 不改底层数据，只改 stride（视图操作） | 通常也不拷贝，但要求内存连续 |
| **常见用途** | 调整维度顺序，如 `(B, S, H, D) → (B, H, S, D)` | 合并/拆分维度，如 `(B, H, S, D) → (B, S, H*D)` |

**典型组合拳**（多头 Attention 的拆分与合并）：

```python
# 输入: (batch, seq, d_model)
# 目标: (batch, nhead, seq, d_k)  其中 d_k = d_model / nhead

# Step 1: view 拆分最后一个维度
x = x.view(batch, seq, nhead, d_k)
# → (batch, seq, nhead, d_k)

# Step 2: transpose 交换维度，让 nhead 提前
x = x.transpose(1, 2)
# → (batch, nhead, seq, d_k)

# 计算完 Attention 后，合并回来：
# Step 3: transpose 先换回 (batch, seq, nhead, d_k)
x = x.transpose(1, 2).contiguous()  # ← contiguous() 必须加！

# Step 4: view 合并最后两个维度
x = x.view(batch, seq, d_model)
```

**为什么需要 `contiguous()`？**

`transpose` 只改 stride，不改内存布局。转完后内存变成"不连续"的：

```
原始内存: [0,1,2,3,4,5,6,7]          （连续）
transpose后逻辑: [0,4,1,5,2,6,3,7]   （不连续）
底层内存还是: [0,1,2,3,4,5,6,7]，通过 stride 跳读实现
```

`view` 要求内存必须连续，所以 `transpose` 后接 `view` 必须先 `.contiguous()`（必要时重新排列内存）。

> **口诀**：`transpose` 换顺序，`view` 改形状；`transpose` 后要 `view`，先调 `contiguous()`。

#### 行列语义：为什么必须是 `Q @ K^T` 而不是 `K @ Q^T`？

矩阵乘法除了要满足"左行=右列"的维度规则，**行列所代表的语义也决定了顺序**：

| 张量 | 最后两维含义 | 说明 |
|------|-------------|------|
| Q | `(seq_q, d_k)` | 每个行向量代表一个 **query token** |
| K^T | `(d_k, seq_k)` | 转置后每个列向量代表一个 **key token** |
| scores = Q @ K^T | `(seq_q, seq_k)` | `scores[i, j]` = 第 i 个 query 对第 j 个 key 的注意力分数 |

对 `scores` 做 `softmax(dim=-1)`，即对**最后一个维度（key 维度）**归一化，含义是：**每个 query 对所有 key 的权重之和为 1**。这是注意力机制的定义。

如果你写反成 `K @ Q^T`，得到的是 `(seq_k, seq_q)`，softmax 后就变成"每个 key 对所有 query 的权重之和为 1"，语义完全颠倒了。

同理，`attn_weights @ V`：
- `attn_weights`: `(seq_q, seq_k)` —— 每个 query 对各 key 的权重
- `V`: `(seq_v, d_k)` 且 `seq_k == seq_v`
- 结果: `(seq_q, d_k)` —— **保留 query 的数量，输出每个 query 加权后的特征**

如果写成 `V @ attn_weights`，输出变成 `(seq, seq)`，完全丢失了 `d_k` 特征维度，也丧失了"query 查 value"的语义。

> **口诀**：写 `@` 时，先检查维度是否匹配（左行=右列），再确认结果的行/列语义是否正确 —— softmax 沿哪个维度做，决定了谁应该出现在行、谁应该出现在列。

---

## 四、残差连接与 LayerNorm（阶段4：Post-LN vs Pre-LN）

### 4.1 Post-LN（原始论文《Attention Is All You Need》）

```
sublayer_output = Sublayer(LayerNorm(x))
output = x + sublayer_output
```

**特点**：
- LayerNorm 在残差分支的**输入处**（先 Norm，再过子层）
- 残差连接加的是原始输入 `x` 和子层输出
- 深层时梯度传播较困难，训练容易不稳定

### 4.2 Pre-LN（后续改进，GPT/LLaMA 等使用）

```
sublayer_output = Sublayer(x)
output = LayerNorm(x + sublayer_output)
```

**特点**：
- LayerNorm 在残差连接的**输出处**（先过子层，再加残差，最后 Norm）
- 另一种常见写法：`x + Sublayer(LayerNorm(x))` —— 注意这和 Post-LN 的区别很微妙！
- 训练更稳定，深层模型收敛更快
- 现代大模型（GPT、LLaMA、BLOOM）普遍采用

### 4.3 残差路径"干净"与"瓶颈"的含义

**残差连接的本质**：让梯度有一条"高速公路"直接回流，不经过任何变换层。

```
Post-LN:  x → [LayerNorm → Attention] → ⊕ → 输出
                              ↑_________|

Pre-LN:   x → [Attention] → ⊕ → LayerNorm → 输出
                    ↑_________|
```

- **瓶颈（Post-LN）**：梯度从输出回传到输入 `x` 时，必须经过 LayerNorm。LayerNorm 会改变梯度的尺度，相当于高速公路上设了收费站。深层网络收费站太多，梯度容易爆炸或消失，训练不稳定。
- **干净（Pre-LN）**：梯度从 `⊕` 可以直接走残差支路回到 `x`，完全不经过 LayerNorm 和 Attention，这条路是"免费的"。LayerNorm 只影响主路径，不影响旁路，深层训练更稳定。

### 4.4 `nn.LayerNorm` 参数详解

```python
nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
```

| 参数 | 含义 |
|------|------|
| `normalized_shape` | **要对哪几个维度做归一化**。通常传入 `(d_model,)`，表示对最后一个维度（特征维度）归一化 |
| `eps` | 防止除以0的小数，默认 `1e-5` |
| `elementwise_affine` | 是否学习缩放 `γ` 和偏移 `β`，默认 `True` |

**关键理解**：
```python
# 输入 x: (batch, seq_len, d_model)
ln = nn.LayerNorm(d_model)  # 等价于 nn.LayerNorm((d_model,))
out = ln(x)  # 对每个 token 的 d_model 个特征单独做归一化
```

- 计算的是**最后一个维度**的均值和方差
- 每个 token 独立计算，token 之间不混合信息
- 输出 shape 和输入完全一样

**为什么不能空着？** PyTorch 必须知道要归一化多少个元素，才能创建 `γ` 和 `β` 的可学习参数（大小等于 `normalized_shape`）。

### 4.5 本实现建议
原始论文用的是 **Post-LN**：
```python
# Attention 分支
attn_out = self.self_attn(self.norm1(x))
x = x + attn_out   # 残差

# FFN 分支
ffn_out = self.ffn(self.norm2(x))
x = x + ffn_out    # 残差
```

注意：很多开源代码实现顺序是 `norm → sublayer → add`，等价于上述流程。

---

## 五、前馈网络（FFN / Feed Forward Network）

### 5.1 结构
原始论文：
```
FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
        = ReLU(x @ W1 + b1) @ W2 + b2
```

- `W1`: `(d_model, 4 * d_model)`  —— 先升维4倍
- `W2`: `(4 * d_model, d_model)`  —— 再降维回原维度

### 5.2 实现
用两个 `nn.Linear` 夹一个激活函数：
```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, 4 * d_model),
    nn.ReLU(),  # 或 nn.GELU()（BERT 用 GELU）
    nn.Linear(4 * d_model, d_model),
    nn.Dropout(dropout)
)
```

---

## 六、与官方实现对比验证（阶段5）

### 6.1 用户疑问
> "线性矩阵难道不是每次随机初始化的么？"

是的，默认每次创建新层，权重都是随机初始化的。要对比，你需要**让两个模型的权重完全一样**。

### 6.2 验证步骤

**方法A：固定随机种子（简单但不够严谨）**
```python
import torch

torch.manual_seed(42)
my_model = DIYTransformer(...)

torch.manual_seed(42)
official_model = nn.TransformerEncoderLayer(...)
```
注意：如果内部初始化逻辑不同，即使 seed 相同，结果也可能不同。

**方法B：直接拷贝权重（推荐）**
```python
# 1. 创建两个模型
my_model = DIYTransformer(d_model=512, nhead=8)
official = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)

# 2. 把官方模型的参数拷贝到你的模型中
# 需要确保两个模型的参数形状和命名一一对应
with torch.no_grad():
    my_model.q_proj.weight.copy_(official.self_attn.in_proj_weight[:512, :])
    # ... 逐一手动对齐每个参数

# 3. 构造相同输入，对比输出
x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
out1 = my_model(x)
out2 = official(x)

diff = torch.abs(out1 - out2).max()
print(f"最大误差: {diff}")
# 如果实现正确，误差应该在 1e-5 ~ 1e-6 级别
```

**方法C：只对比子模块**
如果整体太难对齐，可以分模块验证：
1. 先单独验证你的 `scaled_dot_product_attention` 输出是否正确
2. 再验证整个 `MultiHeadAttention` 层
3. 最后验证完整的 Encoder Layer

---

## 七、实现检查清单

| 模块 | 检查点 |
|------|--------|
| Embedding | 输入是否为 LongTensor？Position Embedding 用的是正弦还是可学习？ |
| Q/K/V 投影 | 是否正确 reshape / transpose 到 `(batch, nhead, seq, d_k)`？ |
| Attention Score | 是否除以 `sqrt(d_k)`？是否做了 mask？softmax 维度是否为 -1？ |
| 多头合并 | transpose 后是否调用了 `.contiguous()`？view 的形状是否正确？ |
| 残差连接 | 输入 `x` 和子层输出的 shape 是否完全一致？ |
| LayerNorm | `nn.LayerNorm(normalized_shape)` 的参数是否传入 `d_model`？ |
| FFN | 是否是两个 Linear？中间维度是否是 `4 * d_model`？ |
| Dropout | 是否在 attention weights 和 FFN 输出后都加了？ |

---

## 八、常见陷阱

1. **LayerNorm 参数**：`nn.LayerNorm(d_model)` 不是 `nn.LayerNorm()`，必须传入要归一化的维度。
2. **矩阵乘法维度**：`q @ k.T` 在 4D tensor 上行不通，要用 `k.transpose(-2, -1)` 交换最后两个维度。
3. **contiguous()**：transpose 后内存不连续，view 前需要 `.contiguous()`。
4. **FFN 不是单层**：很多人误以为 FFN 是一个 Linear，其实是两个。
5. **batch_first**：PyTorch 官方 `nn.MultiheadAttention` 默认 `batch_first=False`，输入是 `(seq, batch, d_model)`，容易混淆。

---

## 九、扩展：如果要实现 Decoder Layer

Decoder 比 Encoder 多一个 **Cross-Attention** 层：
- Query 来自 Decoder 自身的 Masked Self-Attention 输出
- Key / Value 来自 Encoder 的最终输出
- 其余结构（FFN、残差、LayerNorm）与 Encoder 相同

```
Decoder Layer:
    Masked Self-Attention → Add & Norm
    → Cross-Attention（Q from decoder, K/V from encoder） → Add & Norm
    → FFN → Add & Norm
```

---

## 十、常见问题 FAQ（代码实践补充）

### Q1: 正弦位置编码怎么写？

```python
import math

def get_sinusoidal_pe(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (seq, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) *
        (-math.log(10000.0) / d_model)
    )  # (d_model/2,)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, seq, d_model)
```

**记忆技巧**：`position` 是列向量 `(seq, 1)`，`div_term` 是行向量 `(d/2,)`，广播后正好得到 `(seq, d/2)`。

> **`arange` 为什么必须指定 `dtype=torch.float`？**
> 
> `torch.arange` **默认返回 LongTensor（整数）**。只要张量要参与浮点数学运算（`sin/cos/exp/log/除法`），就必须先转 float，否则会出现**整数除法截断**的 bug：
> ```python
> i = torch.arange(0, 10, 2)          # tensor([0, 2, 4, 6, 8])  Long
> result = i / 10                     # tensor([0, 0, 0, 0, 0])  ❌
> 
> i = torch.arange(0, 10, 2, dtype=torch.float)
> result = i / 10                     # tensor([0.0, 0.2, 0.4, 0.6, 0.8])  ✓
> ```
> **经验法则**：任何要进数学函数的计数张量，都写 `dtype=torch.float`。

### Q2: Dropout 应该设多少？

- 原始论文默认 **0.1**。
- 小模型 / 大数据可降至 **0.05**；现代大模型常用 **0.0 ~ 0.1**。
- 建议统一用 `nn.Dropout(p)` 模块，它会自动处理 `train()` / `eval()` 的开关。

### Q3: Mask 怎么传入？在哪加？

**传入**：`forward(self, x, attention_mask=None)`，`attention_mask` 形状 `(batch, seq_len)`，类型 `BoolTensor`，`True` 表示 pad 位置。

**使用位置**：在 `softmax` 之前，把 pad 位置对应的 score 设为 `-inf`：

```python
if attention_mask is not None:
    # (B, S) → (B, 1, 1, S) 广播到 (B, H, S, S)
    scores = scores.masked_fill(
        attention_mask.unsqueeze(1).unsqueeze(2), float('-inf')
    )
```

> Encoder 只需要 **Padding Mask**；Decoder 才需要 **Causal Mask**（上三角遮罩）。

#### `masked_fill` 到底是什么？

`masked_fill(mask, value)`：**把 `mask` 为 `True` 的位置，替换成 `value`。**

```python
scores = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
mask   = torch.tensor([[False, True,  False],
                       [False, False, True]])

scores.masked_fill(mask, float('-inf'))
# [[  1.0,  -inf,   3.0],
#  [  4.0,   5.0,  -inf]]
```

**关键**：`mask` 的形状必须能和原张量 **broadcast**。`scores` 是 `(B, H, S, S)`，`mask` 是 `(B, S)`，所以要用 `unsqueeze(1).unsqueeze(2)` 插入两个维度变成 `(B, 1, 1, S)`，这样才能广播到 `(B, H, S, S)`。

### Q4: FFN 维度容易写反

原始结构是先升维 4 倍，再降回来：

```python
nn.Linear(d_model, 4 * d_model)   # 先升维
nn.ReLU()
nn.Linear(4 * d_model, d_model)   # 再降维
```

### Q5: 残差连接 + LayerNorm 的两种标准写法

**Post-LN（原始论文）**：
```python
attn_out = self.self_attn(self.norm1(x))
x = x + attn_out
```

**Pre-LN（现代模型 GPT/LLaMA）**：
```python
attn_out = self.self_attn(x)
x = self.norm1(x + attn_out)
```

> 两种都可以，关键是不要写成 `x + norm(sublayer(x))` 这种不标准的混合形式。

### Q6: `unsqueeze` 到底在干什么？参数代表什么？

`unsqueeze(dim)`：**在 `dim` 指定的轴上插入一个大小为 1 的维度。**

```python
x = torch.randn(3, 4)    # (3, 4)
x.unsqueeze(0)   # → (1, 3, 4)   在最前面插
x.unsqueeze(1)   # → (3, 1, 4)   在中间插
x.unsqueeze(-1)  # → (3, 4, 1)   在最后插（等价于 unsqueeze(2)）
```

**什么时候用？—— 为了广播 (broadcast)。**

PyTorch 广播规则：从最后一维开始往前比，要么相等，要么其中一个为 1。

```python
scores: (B, H, S, S)   # 4D
mask:   (B, S)         # 2D

# 要把 mask 广播到 scores，需要对齐维度：
mask = mask.unsqueeze(1).unsqueeze(2)  # (B, S) → (B, 1, 1, S)

# 现在对比：
# scores: (B, H, S, S)
# mask:   (B, 1, 1, S)
#              ↑  ↑  ↑
#              这3个维度等于1，可以广播到 H, S, S
```

> **口诀**：`unsqueeze` 是"升维扩维"，只要两个张量形状对不上、又要做逐元素操作（加减乘除、masked_fill），就用它插入 `1` 来对齐。

### Q7: `attn @ V` 的顺序，有没有更直观的理解方式？

有。用**矩阵乘法的"行视角"**来看。

忘掉 batch 和 head，只看最后两维：

```python
attn_weights:  (seq_q, seq_k)   # 每行是一个 query 的注意力分布，行和为 1
V:             (seq_k, d_k)    # 每行是一个 token 的 value 向量
output = attn_weights @ V      # (seq_q, d_k)
```

**线性代数核心定理**：`C = A @ B` 的第 `i` 行等于 `A` 的第 `i` 行乘矩阵 `B`：

$$C[i, :] = A[i,0] \cdot B[0,:] + A[i,1] \cdot B[1,:] + \dots + A[i,n] \cdot B[n,:]$$

套到 attention：

$$\text{output}[i, :] = \text{attn}[i,0] \cdot V[0,:] + \text{attn}[i,1] \cdot V[1,:] + \dots + \text{attn}[i,S] \cdot V[S,:]$$

翻译成人话：

> **第 i 个 query 的输出 = 用 attn[i, :] 这组权重，对 V 的所有行做加权求和。**

这正是注意力机制的定义。如果写成 `V @ attn`，输出变成 `(seq, seq)`，既丢失了 `d_k` 特征维度，语义也变成了"用 V 去加权 attention 的列"，完全不对。

| 运算 | 左边每行是什么 | 右边每行是什么 | 结果每行是什么 |
|------|--------------|--------------|--------------|
| `Q @ K.T` | query token | key token | query 对 key 的分数 |
| `attn @ V` | 权重分布 | value 向量 | 加权聚合后的新向量 |

> **最终口诀**：attention 的两步乘法，都是**"左边当老板，右边当员工"**。老板（query/attn）决定看谁都重要，员工（key/value）提供被看的内容。**老板永远在左边。**

### Q8: `pow` 的第二个参数是向量，怎么理解？

**逐元素运算（element-wise）**。向量 `b` 的每个元素各自当一次指数，返回和 `b` 形状相同的结果：

```python
a = 10000.0
b = torch.tensor([0.0, 0.5, 1.0, 2.0])

torch.pow(a, b)
# = [10000^0.0, 10000^0.5, 10000^1.0, 10000^2.0]
# = [1.0, 100.0, 10000.0, 100000000.0]
```

这和 `sin(x)`、`exp(x)` 一样：传向量进去，对每个元素分别算，返回同样大小的向量。PyTorch 里绝大多数数学函数默认都是逐元素的。

### Q9: 指数公式的负号从哪来？三种写法等价吗？

原始论文：$PE = \sin\left( \frac{pos}{10000^{2i/d}} \right)$

**第一步：除法变乘法**
$$\frac{pos}{10000^{2i/d}} = pos \times \frac{1}{10000^{2i/d}}$$

**第二步：倒数变负指数（高中公式 $a^{-n} = 1/a^n$）**
$$\frac{1}{10000^{2i/d}} = 10000^{-2i/d}$$

**第三步：换底公式（$a^b = e^{b \cdot \ln a}$）**
$$10000^{-2i/d} = \exp\left( -\frac{2i}{d} \times \ln(10000) \right)$$

所以代码里的 `div_term` 就是 $10000^{-2i/d}$，三种写法完全等价：

```python
# 写法1：官方实现用的 exp + log
div_term_1 = torch.exp(i * (-math.log(10000.0) / d_model))

# 写法2：直接幂运算（更易读）
div_term_2 = 10000.0 ** (-i / d_model)

# 写法3：torch.pow
div_term_3 = torch.pow(10000.0, -i / d_model)

assert torch.allclose(div_term_1, div_term_2)  # True
assert torch.allclose(div_term_1, div_term_3)  # True
```

> 选 `exp` 形式主要是工程惯例（原始实现就这么写的），以及 `exp(k*x)` 更容易看出**指数衰减**的本质。自己写项目用 `10000.0 ** (...)` 也完全正确。

### Q10: 怎么从标量公式跨越到向量（Tensor）实现？

核心思维：**先写循环版，再找向量化模式。**

#### 第一步：写最笨的循环版

```python
import math

def get_pe_loop(seq_len, d_model):
    pe = []
    for pos in range(seq_len):
        row = []
        for i in range(d_model // 2):
            angle = pos / (10000 ** (2 * i / d_model))
            row.append(math.sin(angle))
            row.append(math.cos(angle))
        pe.append(row)
    return torch.tensor(pe)
```

#### 第二步：把内层循环向量化

内层循环对每个固定的 `pos`，计算一组 `angle`：

```python
# 对于某个 pos：
angles = pos * [10000^0, 10000^(-2/d), 10000^(-4/d), ...]
```

右边那个列表就是 `div_term`，有 `d_model/2` 个元素。

#### 第三步：把外层循环也向量化

所有 `pos` 一起算，本质是做**外积（outer product）**：

```python
positions = torch.arange(seq_len).unsqueeze(1)   # (seq, 1)
div_term  = torch.tensor([...])                   # (d_model/2,)
angles = positions * div_term                     # 广播 → (seq, d_model/2)
```

#### 完整映射表

| 数学符号 | 代码变量 | 形状 |
|---------|---------|------|
| $pos$ | `position` | `(seq, 1)` |
| $2i$ | `torch.arange(0, d_model, 2)` | `(d_model/2,)` |
| $10000^{-2i/d}$ | `div_term` | `(d_model/2,)` |
| $pos \times 10000^{-2i/d}$ | `position * div_term` | `(seq, d_model/2)` |
| $\sin(\dots)$ | `torch.sin(...)` | 填入偶数列 |
| $\cos(\dots)$ | `torch.cos(...)` | 填入奇数列 |

#### 训练方法：三步法

以后任何公式→PyTorch 都可以这样练：

1. **写循环版**：不管性能，先确保逻辑对
2. **找遍历维度**：哪个变量遍历 batch？seq？feature？
3. **换 arange + 广播**：循环变量 → `torch.arange` → `unsqueeze` 对齐 → 广播运算

**自检技巧**：写完向量化代码后，逐行 `print(shape)`，对照公式检查。

> **本质区别**：标量实现是"一个数对一个数地算"，向量实现是"一批数对一批数地算"。

## 参考

- Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
- PyTorch `nn.TransformerEncoderLayer` 源码

---

### Q11: Pre-LN 和 Post-LN 还是分不清？原始论文到底用哪种？

**一句话区分**：
- **Post-LN**：先做完子层（Attention/FFN），再加残差，**最后 Norm** → `Norm(x + Sublayer(x))`
- **Pre-LN**：先 Norm，再做子层，最后加残差 → `x + Sublayer(Norm(x))`

**原始论文《Attention Is All You Need》用的是 Post-LN**，BERT 也是 Post-LN。  
**GPT-3、LLaMA 等现代大模型用的是 Pre-LN**。

#### 为什么有两种？为什么后来改成了 Pre-LN？

想象训练一个 96 层的 Transformer，梯度要从输出层一路走残差高速公路回到输入层：

**Post-LN 的问题（高速公路上的收费站）**：
```
x → [Attention] → ⊕ → [LayerNorm] → 输出
              ↑______|
```
梯度从输出回传到 `x` 时，必须经过 LayerNorm。LayerNorm 会改变梯度的尺度和分布。层数少（6-12层）时问题不大；层数一多，相当于高速公路上每隔几公里就收一次费，深层梯度要么爆炸要么消失，训练极不稳定。

**Pre-LN 的优势（高速公路免费通行）**：
```
x → [LayerNorm] → [Attention] → ⊕ → 输出
                            ↑______|
```
LayerNorm 被放在了支路（Attention 的输入处），主高速公路（从 ⊕ 直接回到 x 的那条残差边）是**完全干净**的，不经过任何变换。无论网络多深，梯度都能顺畅地从最后一层流回第一层。

#### 记忆口诀

| 类型 | 口诀 | 代表模型 |
|------|------|---------|
| **Post-LN** | "做完再 Norm" | 原始 Transformer、BERT |
| **Pre-LN** | "Norm 了再做" | GPT-3、LLaMA、BLOOM |

> 你的代码当前是 **Post-LN**（`x = self.norm1(x + attn_out)`），和原始论文一致。

---

### Q12: `attention_mask` 到底怎么用？1 是有效还是 mask？

**这是 Transformer 实现中最容易踩的坑之一**，因为不同库有不同约定。

#### 两种主流约定

**约定 A：HuggingFace 风格（推荐）**
```
1 = 有效 token（保留）
0 = padding / mask（需要忽略）
```
对应代码：
```python
scores = scores.masked_fill(attention_mask == 0, float("-inf"))
```
当你用 `tokenizer(..., return_attention_mask=True)` 时，输出的就是这种 mask。

**约定 B：PyTorch 官方风格**
```
True = 需要忽略的位置（mask）
False = 有效位置
```
对应代码：
```python
scores = scores.masked_fill(key_padding_mask, float("-inf"))
```

#### 建议

如果你是**自己学习**，选一种坚持即可，但**一定要在代码注释里写明你的约定**。

如果你是**要对接 HuggingFace 生态**，强烈建议采用约定 A（1=有效，0=mask），这样你的 `forward(attention_mask=...)` 可以直接接收 `tokenizer` 的输出，不需要额外转换。

#### 自检技巧

无论选哪种，写完后问自己一个问题：
> "如果我的序列长度是 3，最后一个 token 是 padding，那 mask 的最后一个值应该是什么？"

- 选约定 A：mask = `[1, 1, 0]`，代码里 `masked_fill(mask == 0, -inf)` → 最后一个被 mask
- 选约定 B：mask = `[False, False, True]`，代码里 `masked_fill(mask, -inf)` → 最后一个被 mask

两种都能 work，关键是**统一**。
