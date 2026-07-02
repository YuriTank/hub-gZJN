import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dk = embed_dim // num_heads
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x):
        # 拆分为多头: [batch_size, seq, num_heads, dk] -> [batch_size, num_heads, seq, dk]
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)

    def forward(self, x):
        batch_size = x.size(0)
        #线性变换
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        #拆分为多头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        #计算相关度分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        # 加权求和
        # [batch_size, num_heads, seq, dk]
        attn_output = torch.matmul(attn_weights, v)
        # 拼接多头并投影回 embed_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # 线性变换
        output = self.out(attn_output)

        return output, attn_weights

# Transformer
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),  # GELU激活
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        attn_output, attn_weights = self.attention(x)
        #多头注意力 + 残差连接
        x = self.norm1(x + self.dropout(attn_output))
        #前馈网络
        ffn_out = self.ffn(x)
        #前馈网络 + 残差连接
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights

if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    embed_dim = 768
    num_heads = 12

    model = Transformer(embed_dim, num_heads)
    test_input = torch.randn(batch_size, seq_len, embed_dim)
    output, attn_weights = model(test_input)

    print(f"输入序列形状: {test_input.shape}")
    print(f"模型输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
