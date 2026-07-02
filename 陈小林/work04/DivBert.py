import torch
import math
from typing import Dict, List, Tuple

"""
基于已有参数，模拟bert计算过程
BERT 单层 Encoder 的完整实现
结构顺序：
    输入 x
        ↓
    Multi-Head Self-Attention
        ↓
    Add (残差连接) : x + attention_output
        ↓
    LayerNorm
        ↓
    Feed-Forward Network (FFN)
        - Linear1: hidden_size → intermediate_size
        - GELU
        - Linear2: intermediate_size → hidden_size
        ↓
    Add (残差连接) : x + ffn_output
        ↓
    LayerNorm
        ↓
    输出
"""
class DivBert:
    def __init__(  self,
        word_embeddings: torch.Tensor,
        position_embeddings: torch.Tensor,
        token_type_embeddings: torch.Tensor,
        embedding_layer_norm_weight: torch.Tensor,
        embedding_layer_norm_bias: torch.Tensor,
        encoder_layers: List[Dict[str, torch.Tensor]],
        pooler_weight: torch.Tensor,
        pooler_bias: torch.Tensor,
        hidden_size: int = 768,
        num_attention_heads: int = 12):

        self.word_embeddings = word_embeddings
        self.position_embeddings = position_embeddings
        self.token_type_embeddings = token_type_embeddings

        self.embedding_norm_weight = embedding_layer_norm_weight
        self.embedding_norm_bias = embedding_layer_norm_bias

        # 模型配置
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.num_layers = len(encoder_layers)
        self.encoders = encoder_layers

        self.pooler_weight = pooler_weight
        self.pooler_bias = pooler_bias


    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        支持batch训练
        """
        batch_size, length = x.shape
        x = self.word_embeddings[x]
        x = x + self.position_embeddings[torch.arange(length)]
        x = x + self.token_type_embeddings[torch.zeros(length, dtype=torch.long)]

        x = self.layerNorm(x, self.embedding_norm_weight, self.embedding_norm_bias)

        #多头注意力
        for encoder in self.encoders:
            x = self.attention(x, encoder)

        cls = x[0]
        cls = cls @ self.pooler_weight.T + self.pooler_bias
        cls = torch.tanh(cls)
        return x, cls


    def attention(self, x, encoder:Dict[str, torch.Tensor])-> torch.Tensor:
        batch_size, length, _ = x.shape
        q = x @ encoder['qw'].T + encoder['qb']
        k = x @ encoder['kw'].T + encoder['kb']
        v = x @ encoder['vw'].T + encoder['vb']

        print(f'q is contiguous:{q.is_contiguous()}')
        print(f'k is contiguous:{k.is_contiguous()}')
        print(f'v is contiguous:{v.is_contiguous()}')

        q = q.reshape(batch_size, length, self.num_attention_heads, self.head_size).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, length, self.num_attention_heads, self.head_size).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, length, self.num_attention_heads, self.head_size).transpose(1, 2).contiguous()

        x_new = torch.softmax(q @ k.transpose(-2,-1) / math.sqrt(self.head_size), dim=-1) @ v
        x_new = x_new.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        x_new = x_new @ encoder['w1'].T + encoder['b1']

        #残差处理
        x = self.layerNorm(x + x_new, encoder['norm1w'], encoder['norm1b'])

        #feed前馈处理
        x_new = self.feedForward(x, encoder)

        #残差处理
        x = self.layerNorm(x + x_new, encoder['norm2w'], encoder['norm2b'])

        return x

    def feedForward( self, x: torch.Tensor, encoder:Dict) -> torch.Tensor:
        x = x @ encoder['feed_w1'].T + encoder['feed_b1']
        #激活
        x = torch.nn.functional.gelu(x)
        x = x @ encoder['feed_w2'].T + encoder['feed_b2']
        return x

    def layerNorm(self, x, weight, bias):
        """归一化"""
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - x_mean) / (x_std + 1e-12)
        return weight * x + bias






