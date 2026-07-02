"""
week3 作业：基于文本的多分类任务
任务规则：5个字的文本中包含"你"，"你"在第几位就属于第几类（0-4）
分别使用 RNN、LSTM 模型完成训练
"""

import torch
import torch.nn as nn
import numpy as np
import random

random.seed(42)
torch.manual_seed(42)

# ─── 配置 ───────────────────────────────────────────────────
CHAR_POOL = "我他她的了在是有不这人们来到上大中国时个和说要就出会也对能下过子得着里去后面"
VOCAB = {"<pad>": 0, "你": 1}
for ch in CHAR_POOL:
    VOCAB[ch] = len(VOCAB)
VOCAB_SIZE = len(VOCAB)

EMBED_DIM = 20
HIDDEN_DIM = 32
SEQ_LEN = 5
NUM_CLASSES = 5
EPOCHS = 20
BATCH_SIZE = 64
TRAIN_SAMPLE = 5000

# ─── 数据生成 ───────────────────────────────────────────────
def build_dataset(sample_num):
    X, Y = [], []
    for _ in range(sample_num):
        pos = random.randint(0, SEQ_LEN - 1)
        text = [random.choice(CHAR_POOL) for _ in range(SEQ_LEN)]
        text[pos] = "你"
        X.append([VOCAB[ch] for ch in text])
        Y.append(pos)
    return torch.LongTensor(X), torch.LongTensor(Y)

# ─── 模型 ───────────────────────────────────────────────────
class TextClassifier(nn.Module):
    def __init__(self, model_type="rnn"):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        if model_type == "rnn":
            self.encoder = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        else:
            self.encoder = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, _ = self.encoder(x)
        logits = self.fc(output[:, -1, :])
        if y is not None:
            return self.loss_fn(logits, y)
        return logits

# ─── 训练 ───────────────────────────────────────────────────
def train(model_type):
    print(f"\n{'='*50}\n  模型: {model_type.upper()}\n{'='*50}")
    model = TextClassifier(model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_x, train_y = build_dataset(TRAIN_SAMPLE)
    test_x, test_y = build_dataset(200)

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for i in range(TRAIN_SAMPLE // BATCH_SIZE):
            x = train_x[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            y = train_y[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(test_x), dim=1)
            acc = (pred == test_y).float().mean().item()
        print(f"  Epoch {epoch+1:2d}  loss={np.mean(losses):.4f}  acc={acc:.4f}")

if __name__ == "__main__":
    train("rnn")
    train("lstm")
