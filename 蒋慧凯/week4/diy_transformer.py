#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动实现一层transfomer
@author: jianghuikai
@date: 2026/05/11
"""

import math
from re import S
from typing import Optional

import torch
from torch import nn
from loguru import logger


class TransformerEncoderLayer(nn.Module):
    """
    transformer-encoder
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        n_header: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = max_seq_len
        self.n_header = n_header
        self.dropout = dropout
        # 多头注意力层
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.zcat = nn.Linear(d_model, d_model)
        self.laynorm1 = nn.LayerNorm(d_model)
        self.laynorm2 = nn.LayerNorm(d_model)
        # FFN层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 多头注意力计算 pre-laynorm 1 计算前先归一化
        attn_out = self.multiheader_attention(self.laynorm1(x), attention_mask)  #
        x = x + attn_out
        # ffn
        ffn_out = self.ffn(self.laynorm2(x))
        x = x + ffn_out
        return x

    def multiheader_attention(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.Q(x)  # (batch,seq_len,d)
        k = self.K(x)  # (batch,seq_len,d)
        v = self.V(x)  # (batch,seq_len,d)
        # 按照每个头切片
        batch, seq_len, d = q.shape
        n_header = self.n_header
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
            # 掩码位置设置为无穷小 比如pad_mask （batch,seq）需要先广播对齐
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # batch,1,1,seq
            attention = attention.masked_fill(attention_mask, float("-inf"))
        attention = torch.softmax(attention, -1)
        attention = self.attn_dropout(attention)
        out = attention @ v  # (batch,n_header,seq_len,d) # 融合
        # 重新拼接
        out: torch.Tensor = out.transpose(1, 2)  # (batch,seq_len,h_header,dk)
        out = out.contiguous().view(batch, seq_len, d)  # (batch,seq_len,d)
        # 线性映射
        out = self.zcat(out)
        return out


class DIYOriginalTransformer(nn.Module):
    """
    原始Transformer架构
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, d_model: int = 768, vocab_size: int = 35536, max_seq_len: int = 512
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model, max_seq_len)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        token_embedding = self.token_embedding(x)  # batch,seq,d
        position_embedding = self.get_position_encodding(x.shape[1])  # 1,seq,d
        x = token_embedding + position_embedding  # 融合，广播运算， batch,seq,d
        x = self.encoder_layer(x, attention_mask)
        return x

    def get_position_encodding(self, seq_len: int) -> torch.Tensor:
        """
        正余弦编码 按照原始论文实现
        """
        d_model = self.d_model
        pe = torch.zeros(seq_len, d_model)  # （seq_len，d_model）
        # 按照位置编码pos、向量维度i初始化
        # 位置id
        pos_idx = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len,1)
        # 词向量dmodel
        div_term = torch.pow(
            10000.0, -torch.arange(0, d_model, 2, dtype=torch.float) / d_model
        ).unsqueeze(0)  # 1,d_model//2
        # 广播计算
        pe[:, 0::2] = torch.sin(pos_idx * div_term)  # (seq_len,d_model)
        pe[:, 1::2] = torch.cos(pos_idx * div_term)  # (seq_len,d_model)
        return pe.unsqueeze(0)  # (1,seq_len,d_model)


class DIYBERTTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 35536,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.seg_embedding = nn.Embedding(2, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        seg: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        token_embedding = self.token_embedding(x)  # batch,seq,d
        seq_len = x.shape[1]
        position_idx = torch.arange(0, seq_len, dtype=torch.int32).unsqueeze(0)  # 1,seq
        position_embedding = self.position_embedding(position_idx)  # 1,seq,d
        seg_embedding = self.seg_embedding(seg)
        x = (
            token_embedding + position_embedding + seg_embedding
        )  # 融合，广播运算， batch,seq,d
        x = self.encoder_layer(x, attention_mask)
        return x


if __name__ == "__main__":
    batch_size = 100
    seq_len = 512
    d_model = 768
    vocab_size = 35536
    encoder = DIYOriginalTransformer(
        d_model=d_model, max_seq_len=seq_len, vocab_size=vocab_size
    )
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    y = encoder(x)
    assert y.shape == (batch_size, seq_len, d_model)  # 形状相同

    logger.add(f"{__file__}.log", mode="w", encoding="utf8", level="DEBUG")
    logger.debug(f"手动实现原始Transformer输出形状:{y.shape}")

    encoder = DIYBERTTransformer(
        d_model=d_model, max_seq_len=seq_len, vocab_size=vocab_size
    )
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    seg = torch.zeros(size=(batch_size, seq_len), dtype=torch.int32)
    y = encoder(x, seg)
    assert y.shape == (batch_size, seq_len, d_model)  # 形状相同
    logger.debug(f"手动实现BERT层Transformer输出形状:{y.shape}")
