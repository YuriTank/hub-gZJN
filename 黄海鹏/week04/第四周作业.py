import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

'''
设计一个单层 transformer Block

Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "不能整除"
        self.d_k = d_model // num_heads

        # 线性变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()
        # 1. 线性投影并拆分多头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V= self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attention_weight = torch.softmax(scores, dim=-1)
        # 3.加权求和
        context = torch.matmul(attention_weight, V)

        # 4. 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5 输出投影
        output = self.w_o(context)

        return output
# 对序列中每个位置独立应用，两层线性 + GELU 激活
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden_size)
        self.linear2 = nn.Linear(ffn_hidden_size, d_model)
        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden_size=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden_size, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        att_output = self.attention(x, x, x)
        z = self.norm1(self.dropout(att_output) + x)

        ffn_output = self.ffn(z)
        output = self.norm2(self.dropout(ffn_output) + z)
        return output
    
if __name__ == '__main__':
    batch_size = 5
    seq_len = 5
    num_heads = 8
    d_model = 512    
    # 随机输入x
    x = torch.randn(batch_size, seq_len, d_model)
    
    model = TransformerBlock(d_model, num_heads)

    output = model(x)

    print(f'输入x的形状: {x.shape}')
    print(f'输出x的形状: {output.shape}')

