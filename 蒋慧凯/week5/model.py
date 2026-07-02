#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动实现一层transfomer
@author: jianghuikai
@date: 2026/05/11
"""

import math
from typing import Optional, List

import torch
from torch import nn, Tensor


class TransformerEncoderLayer(nn.Module):
    """
    transformer-encoder
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        d_model: int,
        n_head: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        # 多头注意力层 #
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(self.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_value: int = 1,
    ):
        attn_out = self.multiheader_attention(x, attention_mask, mask_value)  #
        # post-LN
        x = self.norm1(x + attn_out)
        # ffn
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

    def multiheader_attention(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_value: int = 1,
    ) -> torch.Tensor:
        """多头注意力计算"""
        q = self.Q(x)  # (batch,seq_len,d)
        k = self.K(x)  # (batch,seq_len,d)
        v = self.V(x)  # (batch,seq_len,d)
        # 按照每个头切片
        batch, seq_len, d = q.shape
        n_header = self.n_head
        dk = d // n_header
        q = q.view(batch, seq_len, n_header, dk).transpose(
            1, 2
        )  # (batch,n_header,seq_len,dk)
        k = k.view(batch, seq_len, n_header, dk).transpose(1, 2)
        v = v.view(batch, seq_len, n_header, dk).transpose(1, 2)
        # Attention(Q,K)V 广播运算
        attention: torch.Tensor = (
            q @ k.transpose(-1, -2) / math.sqrt(dk)
        )  # (batch,n_header,seq_len,seq_len)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # batch,1,seq,seq
            attention = attention.masked_fill(
                attention_mask == mask_value, float("-inf")
            )
        attention = torch.softmax(attention, -1)
        attention = self.attn_dropout(attention)
        out = attention @ v  # (batch,n_header,seq_len,d) # 融合
        # 重新拼接
        out: torch.Tensor = out.transpose(1, 2)  # (batch,seq_len,h_header,dk)
        out = out.contiguous().view(batch, seq_len, d)  # (batch,seq_len,d)
        # 线性映射
        out = self.out_proj(out)
        return out


class GPTModel(nn.Module):
    """GPT自回归模型"""

    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 35536,
        max_seq_len: int = 512,
        n_layers: int = 6,
        n_head: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.encoder_layer = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model=d_model, n_head=n_head)
                for _ in range(n_layers)
            ]
        )
        self.lm_header = nn.Linear(
            d_model, vocab_size, bias=False
        )  # 映射到每个词表概率

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向计算 输出词表的概率
        """
        token_embedding = self.token_embedding(x)  # batch,seq,d
        seq_len = x.shape[1]
        position_idx = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)  # 1,seq
        position_embedding = self.position_embedding(position_idx)  # 1,seq,d
        x = token_embedding + position_embedding  # 融合，广播运算， batch,seq,d
        if padding_mask is not None:
            casual_mask = self.build_casual_mask(seq_len)  # seq,seq
            # 需要广播运算 (batch,seq) | (seq,seq)
            padding_mask = padding_mask.unsqueeze(1)  # batch,1,seq
            casual_mask = casual_mask.unsqueeze(0)  # 1,seq,seq
            mask = padding_mask | casual_mask  # 1是需要遮蔽的部分 batch,seq,seq
        else:
            mask = self.build_casual_mask(seq_len)  # 和已有的padding mask合并

        for layer in self.encoder_layer:
            x = layer(x, mask, 1)  # batch,seq,d

        x = self.lm_header(x)
        return x

    def build_casual_mask(self, seq_len: int) -> torch.Tensor:
        """构建三角矩阵，让每个单词只能看见自己及之前的部分"""
        ones = torch.ones((seq_len, seq_len))
        mask = torch.triu(
            ones, diagonal=1
        )  # 上三角，对角线及以上为1（需要掩码遮蔽），对角线及以下为0(需要保留)
        return mask.long()

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 50, eos_id: int = 3
    ) -> torch.Tensor:
        """生成下一个字符
        input_ids: 1,seq
        """
        max_seq_len = self.position_embedding.num_embeddings
        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= max_seq_len:  # 超过最长序列长度
                break
            # 根据
            logits: torch.Tensor = self.forward(input_ids)  # 1,seq,vocab
            # 取最后一个字符的logits
            next_token_logits = logits[:, -1, :]  # 1,vocab
            next_token_id = torch.argmax(next_token_logits, dim=-1)  # 1
            if next_token_id == eos_id:
                break
            # 不断的预测下一个字符的位置
            input_ids = torch.cat(
                [input_ids, next_token_id.unsqueeze(1)], dim=1
            )  # 在seq维度拼接

        return input_ids


if __name__ == "__main__":
    batch_size = 100
    seq_len = 512
    d_model = 768
    vocab_size = 35536
