'''尝试用pytorch实现一个transformer层。'''
#① z = LayerNorm( x + MHA(x) ) 
#② output = LayerNorm( z + FFN(z) )

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 手动实现多头自注意力 
# ==========================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 768 // 12 = 64
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        T, C = x.shape  # 输入: [128, 768] 
        
        # 1. 线性投影
        q = self.W_Q(x)  # [128, 768]
        k = self.W_K(x)  # [128, 768]
        v = self.W_V(x)  # [128, 768]
        
        # 2. 拆分多头 
        q = q.view(T, self.num_heads, self.d_k)  # [128, 12, 64]
        k = k.view(T, self.num_heads, self.d_k)  # [128, 12, 64]
        v = v.view(T, self.num_heads, self.d_k)  # [128, 12, 64]
        
        # 将 "头" 维度换到最前面，让每个头独立处理整个序列
        q = q.permute(1, 0, 2)  # [12, 128, 64]  ← 12个头，每个头是 [128, 64]
        k = k.permute(1, 0, 2)  # [12, 128, 64]
        v = v.permute(1, 0, 2)  # [12, 128, 64]
        
        # 3. 计算注意力得分
        # Q(12, 128, 64) × K^T(12, 64, 128) = scores(12, 128, 128)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # [12, 128, 128]
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. 注意力权重乘以 V
        # weights(12, 128, 128) × V(12, 128, 64) = context(12, 128, 64)
        context = torch.matmul(attn_weights, v)  # [12, 128, 64]
        
        # 5. 拼接多头：变回原来的形状
        context = context.permute(1, 0, 2).contiguous()  # [128, 12, 64]
        context = context.view(T, C)  # [128, 768]  ← 12个头重新拼成 768
        
        # 6. 输出投影
        out = self.W_O(context)  # [128, 768]
        return out

# ==========================================
# 2. 实现 Transformer 层
# ==========================================
class diyTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        #定义 ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),    
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
   
    def forward(self, x):
        # 多头注意力 + 残差 + LayerNorm
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)
        
        # 前馈神经网络 + 残差 + LayerNorm
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x

# ==========================================
# 3. 测试代码 
# ==========================================
if __name__ == '__main__':
    seq_len = 128      
    d_model = 768       
    num_heads = 12      
    d_ff = 3072      
    
    # 生成随机输入 [128, 768]
    x = torch.randn(seq_len, d_model)
    print(x)
    
    # 初始化 Transformer 层
    transformer_layer = diyTransformer(d_model, num_heads, d_ff)
    
    # 前向传播
    output = transformer_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(transformer_layer)
