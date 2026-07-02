
import math

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):

    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward_layer_norm = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, x, attention_mask=None, return_attention_probs=False):
        attention_output, attention_probs = self.self_attention(x, attention_mask)
        x = self.attention_layer_norm(x + self.output_dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.feed_forward_layer_norm(x + self.feed_forward_dropout(feed_forward_output))

        if return_attention_probs:
            return x, attention_probs
        return x

    def self_attention(self, x, attention_mask=None):
        query = self.transpose_for_scores(self.query(x))
        key = self.transpose_for_scores(self.key(x))
        value = self.transpose_for_scores(self.value(x))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = self.build_attention_mask(attention_mask, attention_scores)
            attention_scores = attention_scores.masked_fill(~attention_mask, torch.finfo(attention_scores.dtype).min)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()
        batch_size, seq_len = x.shape[0], x.shape[1]
        context = context.view(batch_size, seq_len, self.hidden_size)
        attention_output = self.attention_output(context)
        return attention_output, attention_probs

    def transpose_for_scores(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)

    @staticmethod
    def build_attention_mask(attention_mask, attention_scores):
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() != 4:
            raise ValueError("attention_mask 只支持 2、3、4 维")

        return attention_mask.to(device=attention_scores.device).bool()


if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    hidden_size = 768

    x = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ])

    transformer_layer = TransformerLayer(hidden_size=hidden_size)
    output, attention_probs = transformer_layer(x, attention_mask, return_attention_probs=True)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("注意力权重形状:", attention_probs.shape)
