import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """ 
    Math:
        Q = X · W_Q,  K = X · W_K,  V = X · W_V
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        """
        
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # 定义三个线性投影矩阵W_Q, W_K, W_V
        # 将输入X投影到三个不同语义子空间, 承担不同角色内容身份
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

        # 输出矩阵的投影
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape # 输入矩阵的batch, B是句子个数, N是每个句子的Token, 每个Token被表示的维度向量

        # 多头自注意力拆成 12 个头，每个头只负责 64 维，让不同的头学习不同的语义关系
        Q = self.W_Q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算得分即: scores = Q · K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # softmax归一化 每一行的注意力分布
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权聚合
        context = torch.matmul(attn, V)

        # 将12个头拼接
        context = context.transpose(1, 2).contiguous().view(B, N, D)

        # 输出投影
        out = self.W_O(context)
        out = self.proj_dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 3072, dropout: float = 0.1):
        super().__init__()

        # 第一层:升维 D → D_ff,扩大特征空间
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # 第二层:降维 D_ff → D,投影回原维度
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 公式: FFN(x) = W_2 · Dropout(GELU(W_1·x + b_1)) + b_2
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class TransFormerBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 ffn_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ------- 子层 1: 多头自注意力 + 残差 + 归一化 -------
        # 公式: x = LayerNorm( x + Attention(x) )
        # 残差项 x 保证梯度可直达浅层,LayerNorm 稳定数值分布
        x = self.norm1(x + self.attn(x))

        # ------- 子层 2: 前馈网络 + 残差 + 归一化 -------
        # 公式: y = LayerNorm( x + FFN(x) )
        x = self.norm2(x + self.ffn(x))
        return x


if __name__ == "__main__":
    EMBEDDING = 768        # 嵌入维度 D
    NUM_HEADS = 12         # 多头数 h,每头维度 d_k = 768/12 = 64
    SEQ_LEN = 16           # 序列长度 N 
    BATCH = 2              # 批大小 B

    # 构造随机输入张量 X ∈ R^(B × N × D)
    x = torch.randn(BATCH, SEQ_LEN, EMBEDDING)
    block = TransFormerBlock(embed_dim=EMBEDDING, num_heads=NUM_HEADS)
    y = block(x)
    print(f"input : {x.shape}")
    print(f"output: {y.shape}")
