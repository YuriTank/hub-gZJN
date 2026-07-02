"""
基于Transformer的字符级语言模型（单向），支持训练和生成。
用法:
    python lm.py --epochs 20                # 训练
    python lm.py --generate "Hello"         # 生成（需已有模型）
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------- 数据 -----------------------------
def load_corpus(pattern="*.txt"):
    texts = [open(p, encoding='utf-8', errors='ignore').read() for p in glob.glob(pattern)]
    return "".join(texts)

def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c:i for i,c in enumerate(chars)}
    idx2char = {i:c for c,i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor([char2idx[c] for c in text if c in char2idx], dtype=torch.long)
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+1:idx+self.seq_len+1]

# ----------------------------- Transformer模型 -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        mask = self._causal_mask(x.size(1), x.device)
        x = self.transformer(x, mask=mask)
        return self.fc(self.dropout(x))

# ----------------------------- 训练 / 评估 -----------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)

# ----------------------------- 文本生成 -----------------------------
@torch.no_grad()
def generate(model, start, char2idx, idx2char, max_new, temperature=0.8, device='cpu'):
    model.eval()
    ids = [char2idx.get(c, 0) for c in start]
    input_ids = torch.tensor([ids], device=device)
    generated = list(start)
    for _ in range(max_new):
        logits = model(input_ids)[0, -1] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated.append(idx2char[next_id])
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
    return ''.join(generated)

# ----------------------------- 主函数 -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--corpus", default="*.txt")
    p.add_argument("--save", default="best_model.pt")
    p.add_argument("--generate", type=str, default=None, help="种子文本")
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 数据
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到txt文件")
    print(f"语料字符数: {len(text):,}")
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_loader = DataLoader(CharDataset(train_text, char2idx, args.seq_len),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(CharDataset(val_text, char2idx, args.seq_len),
                            batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = TransformerLM(vocab_size, args.d_model, args.nhead, args.num_layers, args.d_ff, args.dropout).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ppl = float('inf')
    print(f"\n{'Epoch':>6}  {'Train PPL':>10}  {'Val PPL':>10}")
    print("-" * 32)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if va_ppl < best_ppl:
            best_ppl = va_ppl
            torch.save({'model_state': model.state_dict(), 'char2idx': char2idx, 'idx2char': idx2char}, args.save)
        print(f"{epoch:>6}  {tr_ppl:>10.2f}  {va_ppl:>10.2f}")

    print(f"\n最佳验证PPL: {best_ppl:.2f} 保存至 {args.save}")

    # 生成演示
    if args.generate is not None:
        print("\n生成结果:")
        seed = args.generate.strip() or list(char2idx.keys())[0]
        out = generate(model, seed, char2idx, idx2char, args.max_new, args.temperature, device)
        print(f"种子: {seed}\n{out}")
    else:
        # 简单示例
        seed = list(char2idx.keys())[0] * 3
        out = generate(model, seed, char2idx, idx2char, 100, 0.7, device)
        print(f"\n示例生成 (种子: {seed}):\n{out}")

if __name__ == "__main__":
    main()
