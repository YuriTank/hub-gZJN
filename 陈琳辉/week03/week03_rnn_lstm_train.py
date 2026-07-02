"""
week03_rnn_lstm_train.py
中文酒店评论二分类 —— RNN / LSTM 模型训练

任务：正面/负面评论分类
模型选择：RNN / LSTM（通过 --model 参数切换）
数据集：ChnSentiCorp_htl_all.csv（酒店评论）

用法：
    python week03_rnn_lstm_train.py --model rnn
    python week03_rnn_lstm_train.py --model lstm
    python week03_rnn_lstm_train.py --model lstm --epochs 30 --hidden 128
"""

import os
import re
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jieba

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

jieba.setLogLevel(jieba.logging.INFO)

# ─── 超参数 ────────────────────────────────────────────────────────────────
SEED = 42
MAXLEN = 128
EMBED_DIM = 128
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1

# ─── 随机种子 ─────────────────────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ─── 1. 数据加载与预处理 ──────────────────────────────────────────────────
def load_csv_data(csv_path):
    """加载CSV数据集"""
    texts, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 处理CSV中可能存在的引号
            parts = line.split(',', 1)
            if len(parts) == 2:
                label = int(parts[0])
                text = parts[1].strip('"').strip()
                texts.append(text)
                labels.append(label)
    return texts, labels


def simple_tokenize(text, maxlen=MAXLEN):
    """基于jieba的分词：词语级别分割"""
    words = jieba.lcut(text)
    if len(words) > maxlen:
        words = words[:maxlen]
    return words


def build_vocab(texts, min_freq=2):
    """构建词语级别词表"""
    freq = {}
    for text in texts:
        words = jieba.lcut(text)
        for w in words:
            freq[w] = freq.get(w, 0) + 1

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for w, count in freq.items():
        if count >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode(text, vocab, maxlen=MAXLEN):
    """将文本编码为token id序列"""
    words = simple_tokenize(text, maxlen)
    ids = [vocab.get(w, vocab['<UNK>']) for w in words]
    ids += [vocab['<PAD>']] * (maxlen - len(ids))
    return ids


# ─── 2. Dataset / DataLoader ──────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.X = [encode(t, vocab) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )


# ─── 3. 模型定义 ─────────────────────────────────────────────────────────
class BaseTextRNN(nn.Module):
    """
    基础文本分类模型
    架构：Embedding → RNN/LSTM → MaxPool → Dense → Output
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=1, dropout=DROPOUT, model_type='rnn'):
        super().__init__()
        self.model_type = model_type

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if model_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
            rnn_out_dim = hidden_dim * 2  # 双向RNN
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
            rnn_out_dim = hidden_dim * 2  # 双向LSTM
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.bn = nn.BatchNorm1d(rnn_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_out_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # (B, L, E)

        if self.model_type == 'lstm':
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        # Max Pooling over sequence dimension
        pooled = output.max(dim=1)[0]  # (B, hidden*2)

        pooled = self.dropout(self.bn(pooled))
        out = torch.sigmoid(self.fc(pooled).squeeze(1))
        return out


# ─── 4. 训练与评估函数 ────────────────────────────────────────────────────
class LabelSmoothingBCELoss(nn.Module):
    """带标签平滑的BCE Loss"""
    def __init__(self, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, target_smooth).mean()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds, targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds.extend((pred > 0.5).cpu().numpy())
        targets.extend(y.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return total_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()
            preds.extend((pred > 0.5).cpu().numpy())
            targets.extend(y.cpu().numpy())

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average='binary', zero_division=0
    )
    return total_loss / len(loader), acc, precision, recall, f1


# ─── 5. 可视化 ────────────────────────────────────────────────────────────
def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


# ─── 6. 主训练流程 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='RNN/LSTM Text Classification Training')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['rnn', 'lstm'],
                       help='模型类型: rnn 或 lstm')
    parser.add_argument('--data', type=str,
                       default='',
                       help='数据集路径（默认使用脚本同目录下的ChnSentiCorp_htl_all.csv）')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批次大小')
    parser.add_argument('--hidden', type=int, default=HIDDEN_DIM, help='隐藏层维度')
    parser.add_argument('--embed', type=int, default=EMBED_DIM, help='Embedding维度')
    parser.add_argument('--lr', type=float, default=LR, help='学习率')
    parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout比率')
    parser.add_argument('--maxlen', type=int, default=MAXLEN, help='最大序列长度')
    parser.add_argument('--num_layers', type=int, default=1, help='RNN/LSTM层数')
    parser.add_argument('--output_dir', type=str,
                       default='output',
                       help='输出目录')
    parser.add_argument('--save_model', action='store_true', help='保存模型')
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO,
                       help=f'训练集比例 (默认: {TRAIN_RATIO})')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备选择
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\n{'='*60}")
    print(f"设备: {device}")
    print(f"模型类型: {args.model.upper()}")
    print(f"{'='*60}\n")

    # 1. 加载数据
    print("正在加载数据集...")
    # 处理相对路径
    if args.data:
        csv_path = args.data
    else:
        # 默认使用脚本同目录下的CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'ChnSentiCorp_htl_all.csv')
    texts, labels = load_csv_data(csv_path)
    print(f"  数据集大小: {len(texts)}")
    print(f"  正面评论: {sum(labels)} | 负面评论: {len(labels) - sum(labels)}")

    # 2. 构建词表
    print("\n正在构建词表...")
    vocab = build_vocab(texts, min_freq=2)
    print(f"  词表大小: {len(vocab)}")

    # 3. 数据划分
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=1-args.train_ratio, random_state=SEED,
        stratify=labels
    )
    print(f"\n训练集: {len(train_texts)} | 验证集: {len(val_texts)}")

    # 4. 创建DataLoader
    train_dataset = ReviewDataset(train_texts, train_labels, vocab)
    val_dataset = ReviewDataset(val_texts, val_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # 5. 模型初始化
    model = BaseTextRNN(
        vocab_size=len(vocab),
        embed_dim=args.embed,
        hidden_dim=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {total_params:,} | 可训练: {trainable_params:,}")

    # 6. 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 7. 训练循环
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_acc = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print("开始训练...")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            if args.save_model:
                model_path = os.path.join(args.output_dir, f'best_{args.model}_model.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab': vocab,
                    'args': args,
                    'best_val_acc': best_val_acc
                }, model_path)
                print(f"  [保存最佳模型] Val Acc: {best_val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"{'='*60}")

    # 8. 保存训练曲线
    plot_path = os.path.join(args.output_dir, f'{args.model}_training_curves.png')
    plot_training_curves(history, plot_path)

    # 9. 保存训练历史
    history_path = os.path.join(args.output_dir, f'{args.model}_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # 10. 推理测试
    print(f"\n{'='*60}")
    print("推理测试...")
    print(f"{'='*60}\n")

    model.eval()
    test_samples = [
        "房间非常干净，服务员态度很好，设施齐全，下次还会来",
        "位置很好，交通便利，早餐也不错，推荐入住",
        "房间太小，卫生条件差，前台服务态度恶劣",
        "性价比不高，房间设施老旧，性价比一般",
        "酒店环境优雅，床很舒适，睡眠质量很高"
    ]

    print("测试样本预测结果:")
    print("-" * 60)
    with torch.no_grad():
        for text in test_samples:
            ids = torch.tensor([encode(text, vocab, args.maxlen)], dtype=torch.long).to(device)
            prob = model(ids).item()
            label = "正面" if prob > 0.5 else "负面"
            confidence = prob if prob > 0.5 else 1 - prob
            print(f"文本: {text[:40]}...")
            print(f"预测: {label} (置信度: {confidence:.2%})")
            print("-" * 60)


if __name__ == '__main__':
    main()
