import torch
import torch.nn as nn
import math


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))

        attn_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, V)

        return output


class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super().__init__()

        self.attention = SingleHeadSelfAttention(hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, hidden_size)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)

        # 1. Self-Attention
        attn_output = self.attention(x)

        # 2. Residual + LayerNorm
        x = self.norm1(x + attn_output)

        # 3. Feed Forward
        ff_output = self.ffn(x)

        # 4. Residual + LayerNorm
        x = self.norm2(x + ff_output)

        return x
