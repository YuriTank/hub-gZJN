import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.hidden_size = hidden_size
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        # x: (batch, seq_len, hidden_size) -> (batch, num_heads, seq_len, head_size)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        q = self.transpose_for_scores(self.q(x))  # (batch, num_heads, seq_len, head_size)
        k = self.transpose_for_scores(self.k(x))
        v = self.transpose_for_scores(self.v(x))
        # attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        # context: (batch, num_heads, seq_len, head_size)
        context = torch.matmul(attention_weights, v)
        # concat heads: (batch, seq_len, hidden_size)
        context = context.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, self.hidden_size)
        return self.output(context)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.dense2(self.gelu(self.dense1(x)))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        # self-attention + residual + layer norm
        attention_output = self.attention(x, mask)
        x = self.layer_norm1(x + attention_output)
        # feed forward + residual + layer norm
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x
