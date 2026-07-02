import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AttentionMatrix(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 使用独立的投影层而不是一次计算
        self.query_map = nn.Linear(embed_dim, embed_dim)
        self.key_map = nn.Linear(embed_dim, embed_dim)
        self.value_map = nn.Linear(embed_dim, embed_dim)
        
        # 添加可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def _prepare_heads(self, x, projection):
        B, T, _ = x.shape
        x = projection(x)
        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # [B, n_heads, T, head_dim]
        return x
        
    def forward(self, x_seq, attn_mask=None):
        B, T, D = x_seq.shape
        
        # 生成Q,K,V
        Q = self._prepare_heads(x_seq, self.query_map)
        K = self._prepare_heads(x_seq, self.key_map)
        V = self._prepare_heads(x_seq, self.value_map)
        
        # 计算注意力分数
        scale_factor = 1.0 / self.temperature
        attn_scores = (Q @ K.transpose(-2, -1)) * scale_factor
        
        # 应用掩码
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1) == 0, -1e9)
        
        # 计算注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力
        context = attn_probs @ V
        
        # 合并注意力头
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(B, T, D)
        
        # 输出投影
        return self.output_proj(context), attn_probs


class PositionWiseNetwork(nn.Module):
    def __init__(self, hidden_size, expansion_factor=4):
        super().__init__()
        expanded_size = hidden_size * expansion_factor
        
        # 使用门控机制的前馈网络
        self.gate_proj = nn.Linear(hidden_size, expanded_size)
        self.up_proj = nn.Linear(hidden_size, expanded_size)
        self.down_proj = nn.Linear(expanded_size, hidden_size)
        
        # 添加门控激活
        self.activation = nn.GELU()
        
        # 可选的dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # 门控前馈网络
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        
        # 元素级相乘
        hidden = gate * up
        hidden = self.dropout(hidden)
        
        return self.down_proj(hidden)


class AdaptiveLayerNorm(nn.Module):
    """自适应的LayerNorm，包含可学习的缩放和平移参数"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # 自适应的LayerNorm
        normalized = (x - mean) / (std + self.eps)
        
        # 应用可学习的缩放和平移
        return self.gamma * (self.weight * normalized + self.bias)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_expansion=4):
        super().__init__()
        
        # 注意力层和前馈层
        self.attention = AttentionMatrix(hidden_size, num_heads)
        self.feedforward = PositionWiseNetwork(hidden_size, ff_expansion)
        
        # 使用自适应的LayerNorm
        self.attn_norm = AdaptiveLayerNorm(hidden_size)
        self.ff_norm = AdaptiveLayerNorm(hidden_size)
        
        # 额外的门控残差连接
        self.attn_gate = nn.Linear(hidden_size, hidden_size)
        self.ff_gate = nn.Linear(hidden_size, hidden_size)
        
        # 可选的dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        
        # 注意力子层
        attn_input = self.attn_norm(x)
        attn_output, attn_probs = self.attention(attn_input, attn_mask)
        
        # 门控残差连接
        gate = torch.sigmoid(self.attn_gate(x))
        x = x + gate * self.dropout(attn_output)
        
        # 前馈子层
        ff_input = self.ff_norm(x)
        ff_output = self.feedforward(ff_input)
        
        # 门控残差连接
        gate = torch.sigmoid(self.ff_gate(x))
        x = x + gate * self.dropout(ff_output)
        
        return x, attn_probs


class LearnedPositionEmbedding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, max_len, hidden_size):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        self.pos_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, start_pos=0):
        B, T, D = x.shape
        
        # 获取位置编码
        if start_pos + T > self.pos_embedding.shape[1]:
            # 如果序列长度超过最大长度，使用截断
            pos_emb = self.pos_embedding[:, :T]
        else:
            pos_emb = self.pos_embedding[:, start_pos:start_pos + T]
        
        # 应用位置编码投影
        pos_emb = self.pos_proj(pos_emb)
        
        return x + pos_emb


class TransformerStack(nn.Module):
    def __init__(self, 
                 hidden_dim=768, 
                 num_layers=12, 
                 num_heads=12, 
                 ff_expansion=4,
                 max_seq_len=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 创建多个Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_expansion)
            for _ in range(num_layers)
        ])
        
        # 最终归一化层
        self.final_norm = AdaptiveLayerNorm(hidden_dim)
        
        # 可学习的位置编码
        self.position_embedding = LearnedPositionEmbedding(max_seq_len, hidden_dim)
        
        # 添加可选的输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, attn_mask=None, return_attention=False):
        B, T, D = x.shape
        
        # 应用位置编码
        x = self.position_embedding(x)
        
        attention_maps = []
        
        # 逐层处理
        for i, block in enumerate(self.blocks):
            x, attn_probs = block(x, attn_mask)
            if return_attention:
                attention_maps.append(attn_probs)
        
        # 最终归一化
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        if return_attention:
            return x, attention_maps
        
        return x


class EmbeddingWithDropout(nn.Module):
    """带dropout的嵌入层"""
    def __init__(self, vocab_size, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        # 嵌入层归一化
        self.norm = AdaptiveLayerNorm(hidden_size)
        
    def forward(self, token_ids):
        embeds = self.embedding(token_ids)
        embeds = self.norm(embeds)
        return self.dropout(embeds)


class CompleteTransformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, 
                 num_heads=12, ff_expansion=4, max_seq_len=512):
        super().__init__()
        
        # 词嵌入
        self.token_embedding = EmbeddingWithDropout(vocab_size, hidden_size)
        
        # Transformer编码器堆叠
        self.encoder = TransformerStack(
            hidden_dim=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_expansion=ff_expansion,
            max_seq_len=max_seq_len
        )
        
        # 可选的输出头
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, vocab_size)
        )
        
    def forward(self, input_ids, attention_mask=None, return_attention=False):
        # 获取词嵌入
        x = self.token_embedding(input_ids)
        
        # 创建注意力掩码
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None
        
        # 通过编码器
        if return_attention:
            x, attention_maps = self.encoder(x, attn_mask, return_attention=True)
        else:
            x = self.encoder(x, attn_mask)
        
        # 可选：应用输出头
        if hasattr(self, 'output_head'):
            x = self.output_head(x)
        
        if return_attention:
            return x, attention_maps
        return x


# 工具函数
def create_causal_attention_mask(seq_len, device):
    """创建因果注意力掩码"""
    mask = torch.ones(seq_len, seq_len, device=device)
    mask = torch.triu(mask, diagonal=1).bool()
    return ~mask


def create_padding_mask(input_ids, pad_token_id=0):
    """创建padding掩码"""
    mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = CompleteTransformer(
        vocab_size=10000,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        ff_expansion=4,
        max_seq_len=256
    )
    
    # 测试输入
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    # 创建padding掩码
    # 模拟一些padding tokens
    for i in range(batch_size):
        input_ids[i, seq_len//2:] = 0
    
    attention_mask = (input_ids != 0)
    
    # 前向传播
    output, attention_maps = model(input_ids, attention_mask, return_attention=True)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力图数量: {len(attention_maps)}")
    print(f"每个注意力图形状: {attention_maps[0].shape}")
    
    # 验证模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试推理模式
    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        print(f"\n推理模式输出形状: {output.shape}")
