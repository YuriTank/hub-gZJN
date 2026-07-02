import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape

        # 线性变换
        Q = self.w_q(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # (batch, n_head, seq_len, head_dim)
        K = self.w_k(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, n_head, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (batch, n_head, seq_len, head_dim)

        # 合并多头并输出
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """前馈网络（两层线性层 + ReLU）"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    """一个完整的 Transformer 编码器层"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头自注意力 + 残差 + 层归一化
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # 前馈网络 + 残差 + 层归一化
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# 示例
if __name__ == "__main__":
    batch, seq_len, d_model = 2, 10, 512
    n_head, d_ff = 8, 2048

    layer = TransformerEncoderLayer(d_model, n_head, d_ff)
    x = torch.randn(batch, seq_len, d_model)
    out = layer(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
