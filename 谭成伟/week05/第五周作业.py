"""
字符级语言模型训练脚本，使用 Transformer 实现，含 PPL 计算和文本续写。
用法:
    python language_model.py --epochs 20
    python language_model.py --sample "黄金价格"  # 文本续写
"""

import math
import os
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    # 文本字符去重
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    # 返回 字符对应下标，下标对应字符
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── Transformer 模型 ───────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super().__init__()
        assert hidden % n_head == 0
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3) # 一次性算 Q K V
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        # self.qkv(x) =》[B, T, hidden * 3]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        # 输出: [batch, num_heads, seq_len, seq_len]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_head, ff, dropout):
        super().__init__()
        # 多头注意力 self.attn => [batch, seq_len, hidden_size]
        self.attn = MultiHeadAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.drop(self.attn(x, mask)))
        x = self.ln2(x + self.drop(self.ffn(x)))
        # x:[batch_size, seq_len, hidden_size]
        return x


class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, n_head, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim))
        
        self.layers = nn.ModuleList([
            # hidden_dim * 4 作为transformer模型中前馈层的第一个线性层的出参，作为下一个线性层的入参
            TransformerBlock(hidden_dim, n_head, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.drop = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim


    def forward(self, x):  # 模型的"工作流程"函数，x是输入的字符序列
        B, T = x.shape
        
        e = self.drop(self.embed(x) + self.pos_embed[:, :T])
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        for layer in self.layers:
            e = layer(e, mask)
        
        logits = self.fc(e)
        
        # 返回结果，供后续计算损失或生成文本
        return logits


# ─────────────────────────── top-p 采样 ───────────────────────────

def top_p_sample(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    mask = cumulative_probs <= p
    
    mask[0] = True
    
    sorted_probs = sorted_probs * mask
    
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    idx = torch.multinomial(sorted_probs, num_samples=1)
    
    return sorted_indices[idx]


def generate_text(model, start_text, char2idx, idx2char, max_len=200, top_p=0.9, device="cpu"):
    model.eval()
    input_ids = [char2idx[c] for c in start_text if c in char2idx]
    if not input_ids:
        return "起始文本中没有有效的字符"
    
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_tensor)  # logits形状: [B=1, T=当前句子长度, V=词表大小]
            last_logits = logits[0, -1, :] # [0, -1] 相当于取出最后一个字的所有字的预测概率，[:]相当于取出所有的概率
            next_idx = top_p_sample(last_logits, p=top_p)  # 用top-p采样选择一个字符
            input_tensor = torch.cat([input_tensor, next_idx.unsqueeze(0)], dim=1)
            if next_idx.item() == char2idx.get("\n", -1):
                break
    
    # 把生成的字符索引转换成实际的中文字符
    # input_tensor[0]: 取出第一个样本（因为batch_size=1）
    # idx.item(): 把每个字符的索引从张量变成数字
    # idx2char[idx.item()]: 用索引查找对应的中文字符
    # 比如索引123可能对应"价"字
    generated = [idx2char[idx.item()] for idx in input_tensor[0]]
    
    # 把字符列表拼接成完整的字符串返回
    # ["黄", "金", "价", "格"] -> "黄金价格"
    return "".join(generated)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0
    total_batches = len(loader)

    for batch_idx, (x, y) in enumerate(loader):
        # x =》 [batch_size, seq_len]
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        batch_count += 1
        
        if train and batch_count % 10 == 0:
            avg_loss_so_far = total_loss / total_tokens
            ppl_so_far = math.exp(avg_loss_so_far)
            progress = (batch_count / total_batches) * 100
            print(f"\r  Batch {batch_count}/{total_batches} ({progress:.1f}%) - Loss: {avg_loss_so_far:.4f}, PPL: {ppl_so_far:.2f}", end="")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    if train:
        print()
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seq_len",    type=int,   default=32)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--num_layers", type=int,   default=6)
    parser.add_argument("--n_head",     type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0)
    # parser.add_argument("--lr",         type=float, default=0.08) # 这里0.08 反而比0.1下降的快
    # parser.add_argument("--lr",         type=float, default=0.5) # 这里0.3-0.5出现梯度爆炸
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    parser.add_argument("--sample",     type=str,   default="", help="文本续写的起始文本")
    parser.add_argument("--max_len",    type=int,   default=200, help="续写最大长度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    if args.sample:
        if not os.path.exists(args.save):
            raise FileNotFoundError(f"未找到模型文件 {args.save}")
        checkpoint = torch.load(args.save, map_location=device)
        model_args = checkpoint["args"]
        
        model = LM(
            vocab_size=vocab_size,
            embed_dim=model_args.get("embed_dim", 128),
            hidden_dim=model_args.get("hidden_dim", 128),
            num_layers=model_args.get("num_layers", 3),
            n_head=model_args.get("n_head", 4),
            dropout=model_args.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        
        result = generate_text(model, args.sample, char2idx, idx2char, max_len=args.max_len, device=device)
        print(f"\n起始文本: {args.sample}")
        print(f"续写结果:\n{result}")
        return

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}] 开始训练...")
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        
        print(f"[Epoch {epoch}/{args.epochs}] 开始验证...")
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            print(f"[Epoch {epoch}/{args.epochs}] 找到更好的模型，保存至 {args.save}")
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")
    
    print("\n=== 测试文本续写 ===")
    test_samples = ["黄金价格", "美联储", "欧债危机", "农产品"]
    for sample in test_samples:
        result = generate_text(model, sample, char2idx, idx2char, max_len=50, device=device)
        print(f"\n起始: {sample}")
        print(f"续写: {result}")


if __name__ == "__main__":
    # 训练模式（不传 --sample 参数时）
    # main()

    # 测试模式 - 加载已训练模型进行文本续写
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("best_model.pt", map_location=device)
    model_args = checkpoint["args"]
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=model_args.get("embed_dim", 128),
        hidden_dim=model_args.get("hidden_dim", 128),
        num_layers=model_args.get("num_layers", 3),
        n_head=model_args.get("n_head", 4),
        dropout=model_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 60)
    print("加载模型成功，开始文本续写测试")
    print("=" * 60)

    test_samples = ["黄金价格", "美联储", "欧债危机", "农产品"]
    for sample in test_samples:
        print(f"\n起始文本: {sample}")
        result = generate_text(model, sample, char2idx, idx2char, max_len=100, device=device)
        print(f"续写结果: {result}")
        print("-" * 60)
