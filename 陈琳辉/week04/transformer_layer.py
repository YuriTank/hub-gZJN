"""
手写Transformer层 - Week04作业
===========================

本文件实现了一个完整的Transformer编码器层，包含：
1. 多头自注意力机制（Multi-Head Self-Attention）
2. 前馈神经网络（Feed-Forward Network）
3. 残差连接和层归一化（Residual Connection & Layer Normalization）

Author: Student
Date: Week04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    --------------------
    
    注意力机制的核心思想：让序列中的每个位置都能"关注"到序列中的其他位置。
    通过多头注意力，我们可以从不同的表示子空间捕捉信息。
    
    结构：
    - 输入：Query (Q), Key (K), Value (V)
    - 过程：线性变换 -> 分割成多头 -> Scaled Dot-Product Attention -> 合并多头 -> 线性变换
    - 输出：注意力加权后的表示
    """
    
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力层
        
        参数:
            d_model (int): 模型的维度，即输入输出的特征维度
            num_heads (int): 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        
        # 确保d_model可以被num_heads整除，这样每个头得到相同维度的表示
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model      # 模型维度
        self.num_heads = num_heads  # 注意力头数量
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义Q, K, V的线性变换层
        # 输入输出维度都是d_model，将输入映射到Q, K, V空间
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 最后的输出线性变换层
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        参数:
            query (Tensor): 查询张量，形状 [batch_size, seq_len, d_model]
            key (Tensor): 键张量，形状 [batch_size, seq_len, d_model]
            value (Tensor): 值张量，形状 [batch_size, seq_len, d_model]
            mask (Tensor, optional): 掩码张量，用于遮挡某些位置
        
        返回:
            output (Tensor): 注意力输出，形状 [batch_size, seq_len, d_model]
            attention_weights (Tensor): 注意力权重，形状 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # ========== 第一步：线性变换得到Q, K, V ==========
        # 将输入通过线性层得到查询、键、值向量
        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)    # [batch_size, seq_len, d_model]
        V = self.W_v(value)  # [batch_size, seq_len, d_model]
        
        # ========== 第二步：分割成多个注意力头 ==========
        # 将d_model维度分割成num_heads个d_k维度的头
        # 形状变换：[batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # ========== 第三步：计算注意力分数 ==========
        # 使用scaled dot-product attention
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        # 掩码用于遮挡padding位置或未来位置
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # ========== 第四步：计算注意力权重并加权 ==========
        # 对分数进行softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 用注意力权重对V进行加权求和
        # context = attention_weights @ V
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # ========== 第五步：合并多个注意力头 ==========
        # 形状变换：[batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        # ========== 第六步：最终线性变换 ==========
        output = self.W_o(context)  # [batch_size, seq_len, d_model]
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络（Position-wise Feed-Forward Networks）
    ---------------------------------------------------
    
    这是Transformer中每个位置独立应用的两层全连接网络。
    由两个线性变换组成，中间有一个ReLU激活函数。
    
    结构：Linear -> ReLU -> Linear
    
    公式：FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈神经网络
        
        参数:
            d_model (int): 输入和输出的维度
            d_ff (int): 前馈网络中间层的维度（通常设置为d_model的4倍）
            dropout (float): Dropout比率，用于正则化
        """
        super(FeedForwardNetwork, self).__init__()
        
        # 第一层线性变换：将维度从d_model扩展到d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # 第二层线性变换：将维度从d_ff压缩回d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量，形状 [batch_size, seq_len, d_model]
        
        返回:
            output (Tensor): 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 第一层：线性变换 + ReLU激活
        output = self.linear1(x)  # [batch_size, seq_len, d_ff]
        output = F.relu(output)
        
        # 应用Dropout
        output = self.dropout(output)
        
        # 第二层：线性变换
        output = self.linear2(output)  # [batch_size, seq_len, d_model]
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    ------------------
    
    这是Transformer的基本构建块，包含两个子层：
    1. 多头自注意力层（Multi-Head Self-Attention）
    2. 前馈神经网络层（Feed-Forward Network）
    
    每个子层都使用残差连接（Residual Connection）和层归一化（Layer Normalization）。
    
    结构：
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
    
    残差连接的作用：缓解深层网络的梯度消失问题，让网络更容易训练
    层归一化的作用：稳定训练过程，加速收敛
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化Transformer编码器层
        
        参数:
            d_model (int): 模型的维度
            num_heads (int): 注意力头的数量
            d_ff (int): 前馈网络中间层的维度
            dropout (float): Dropout比率
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头自注意力层
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # 前馈神经网络层
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # 两个层归一化层
        # 注意：第一个用于注意力层输出，第二个用于FFN层输出
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 两个Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量，形状 [batch_size, seq_len, d_model]
            mask (Tensor, optional): 掩码张量
        
        返回:
            output (Tensor): 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # ========== 子层1：多头自注意力 ==========
        # 计算自注意力
        # 注意：在自注意力中，Q、K、V都来自同一个输入
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        
        # 应用Dropout
        attn_output = self.dropout1(attn_output)
        
        # 残差连接 + 层归一化
        # sublayer(x) = LayerNorm(x + SubLayer(x))
        x = self.norm1(x + attn_output)
        
        # ========== 子层2：前馈神经网络 ==========
        # 计算前馈网络输出
        ff_output = self.feed_forward(x)
        
        # 应用Dropout
        ff_output = self.dropout2(ff_output)
        
        # 残差连接 + 层归一化
        # sublayer(x) = LayerNorm(x + SubLayer(x))
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    ----------------
    
    由多个Transformer编码器层堆叠而成。
    每个编码器层独立处理序列，然后将输出传递给下一层。
    
    结构：
    输入 → [EncoderLayer₁] → [EncoderLayer₂] → ... → [EncoderLayerₙ] → 输出
    
    通过堆叠多个编码器层，网络可以学习更复杂的表示。
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化Transformer编码器
        
        参数:
            num_layers (int): 编码器层的数量
            d_model (int): 模型的维度
            num_heads (int): 注意力头的数量
            d_ff (int): 前馈网络中间层的维度
            dropout (float): Dropout比率
        """
        super(TransformerEncoder, self).__init__()
        
        # 创建一个由num_layers个编码器层组成的列表
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量，形状 [batch_size, seq_len, d_model]
            mask (Tensor, optional): 掩码张量
        
        返回:
            output (Tensor): 输出张量，形状 [batch_size, seq_len, d_model]
        """
        # 依次通过每个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# ==================== 测试代码 ====================

def test_transformer():
    """
    测试Transformer编码器的功能
    """
    print("=" * 60)
    print("Transformer层测试")
    print("=" * 60)
    
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    print(f"\n参数配置:")
    print(f"  - 批次大小 (batch_size): {batch_size}")
    print(f"  - 序列长度 (seq_len): {seq_len}")
    print(f"  - 模型维度 (d_model): {d_model}")
    print(f"  - 注意力头数 (num_heads): {num_heads}")
    print(f"  - FFN中间维度 (d_ff): {d_ff}")
    print(f"  - 编码器层数 (num_layers): {num_layers}")
    
    # 创建随机输入
    print(f"\n创建随机输入张量...")
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"  输入形状: {x.shape}")
    
    # 创建掩码（可选）
    mask = None  # 也可以创建padding mask
    
    # ========== 测试单个编码器层 ==========
    print(f"\n" + "-" * 40)
    print("测试单个Transformer编码器层")
    print("-" * 40)
    
    single_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    output_single = single_layer(x, mask)
    print(f"  单层输出形状: {output_single.shape}")
    print(f"  输出统计: 均值={output_single.mean().item():.4f}, 标准差={output_single.std().item():.4f}")
    
    # ========== 测试完整编码器 ==========
    print(f"\n" + "-" * 40)
    print("测试完整Transformer编码器（6层）")
    print("-" * 40)
    
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
    output = encoder(x, mask)
    print(f"  完整编码器输出形状: {output.shape}")
    print(f"  输出统计: 均值={output.mean().item():.4f}, 标准差={output.std().item():.4f}")
    
    # ========== 测试多头注意力 ==========
    print(f"\n" + "-" * 40)
    print("测试多头自注意力机制")
    print("-" * 40)
    
    attention = MultiHeadAttention(d_model, num_heads)
    attn_output, attn_weights = attention(x, x, x, mask)
    print(f"  注意力输出形状: {attn_output.shape}")
    print(f"  注意力权重形状: {attn_weights.shape}")
    print(f"  注意力权重范围: [{attn_weights.min().item():.4f}, {attn_weights.max().item():.4f}]")
    
    # 显示一个头的注意力权重热力图的前几个值
    print(f"  第一个头的注意力权重（第一个样本，第一个头）:")
    print(f"    {attn_weights[0, 0, 0, :5].tolist()}")
    
    # ========== 参数统计 ==========
    print(f"\n" + "-" * 40)
    print("模型参数统计")
    print("-" * 40)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  完整编码器总参数量: {total_params:,}")
    
    params_per_layer = sum(p.numel() for p in single_layer.parameters())
    print(f"  单层参数量: {params_per_layer:,}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    return output


if __name__ == "__main__":
    # 运行测试
    output = test_transformer()
