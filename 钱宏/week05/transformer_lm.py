
import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    注意：为了支持语言模型，这里增加了 causal_mask 的支持
    """
    def __init__(self, hidden_size=768, num_heads=12):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 线性投影 Q K V
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_mask=None, causal_mask=False):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 1. 应用 Padding Mask (如果存在)
        if attention_mask is not None:
            # attention_mask: [batch, seq_len], 1 for valid, 0 for padding
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [batch, 1, 1, seq_len]
            extended_mask = (1.0 - extended_mask.float()) * -10000.0
            scores = scores + extended_mask

        # 2. 应用 Causal Mask (防止看到未来信息)
        if causal_mask:
            # 使用 tril (下三角) 生成掩码
            # torch.tril 生成下三角为1，上三角为0的矩阵
            # diagonal=0 表示包含主对角线（当前 token 可以看到自己）
            causal_mask_matrix = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0).bool()
            
            # 我们需要屏蔽的是“未来”的信息，即下三角掩码中为 0 的位置（上三角部分）
            # masked_fill: 当 mask 为 False 时，填充 -inf
            # 因为 causal_mask_matrix 在下三角是 True，上三角是 False
            # 保留下三角（True），将上三角（False）填充为 -inf
            scores = scores.masked_fill(~causal_mask_matrix.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.W_o(context)
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """单个 Transformer Block"""
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(dropout)

        self.feed_forward = FeedForward(hidden_size, intermediate_size)
        self.ff_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, causal_mask=False):
        # Self-Attention
        attn_output = self.attention(x, attention_mask, causal_mask)
        attn_output = self.attention_dropout(attn_output)
        x = self.attention_norm(x + attn_output)

        # Feed Forward
        ff_output = self.feed_forward(x)
        ff_output = self.ff_dropout(ff_output)
        x = self.ff_norm(x + ff_output)
        
        return x


class BertEmbeddings(nn.Module):
    """BERT 嵌入层"""
    def __init__(self, vocab_size=30522, hidden_size=768, max_position_embeddings=512, dropout=0.1):
        super(BertEmbeddings, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        # 语言模型通常不需要 segment embedding，但为了兼容结构保留，实际使用时可传全0
        self.segment_embedding = nn.Embedding(2, hidden_size) 

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        batch_size, seq_len = input_ids.size()
        token_embeds = self.token_embedding(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        segment_embeds = self.segment_embedding(token_type_ids)

        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerLM(nn.Module):
    """
    基于 Transformer 的语言模型
    架构: Embeddings -> N x TransformerBlock (with Causal Mask) -> Linear Head
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        dropout=0.1
    ):
        super(TransformerLM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # 语言模型头部：将隐藏状态映射回词表大小
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len], optional
        """
        # 1. Embedding
        hidden_states = self.embeddings(input_ids, token_type_ids=None)

        # 2. Transformer Blocks with Causal Mask
        for block in self.transformer_blocks:
            # 关键：开启 causal_mask=True 以实现单向语言模型
            hidden_states = block(hidden_states, attention_mask, causal_mask=True)

        # 3. Final Norm
        sequence_output = self.final_layer_norm(hidden_states)

        # 4. LM Head
        logits = self.lm_head(sequence_output)
        
        return logits


# ==========================================
# 2. 数据处理部分 (复用 language_model.py 的逻辑)
# ==========================================

def load_corpus(pattern="*.txt"):
    """加载符合通配符模式的所有文本文件，合并为一个大字符串。"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)

def build_vocab(text):
    """根据语料文本构建字符到索引、索引到字符的映射表。"""    
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}   
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    """字符级数据集，将文本切分为固定长度的输入-目标序列对。"""
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx] # # 将文本中每个字符转为索引
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        """根据索引获取一个样本：输入序列 x 和目标序列 y（y 比 x 偏移一位）。"""
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

# ==========================================
# 3. 训练与评估部分
# ==========================================

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Transformer 不需要隐藏状态重置，直接前向传播
        logits = model(x) 
        
        # logits shape: [batch, seq_len, vocab_size]
        # criterion expects: [N, C] and [N]
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping is often helpful for Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def main():
    parser = argparse.ArgumentParser()
    # 模型参数调整为适合 Transformer 的默认值
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)    # Transformer 可以处理更长序列，但显存消耗大
    parser.add_argument("--batch_size", type=int,   default=32)    # Transformer 显存占用大，batch size 通常比 RNN 小
    parser.add_argument("--hidden_dim", type=int,   default=256)   # 对应 hidden_size
    parser.add_argument("--num_layers", type=int,   default=4)     # Transformer 层数
    parser.add_argument("--num_heads",  type=int,   default=8)     # 注意力头数
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-4)  # Transformer 通常需要更小的学习率
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer_lm.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型实例化
    model = TransformerLM(
        vocab_size=vocab_size,
        hidden_size=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        intermediate_size=args.hidden_dim * 4, # 前馈网络维度通常是 hidden_dim 的 4 倍
        max_position_embeddings=args.seq_len,  # 位置编码最大长度设为序列长度即可
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

if __name__ == "__main__":
    main()
