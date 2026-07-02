import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

# 设置随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ----------------------------- 1. 字符表与数据生成 -----------------------------
# 为了简化，使用常见汉字（不含“你”作为常规字符）
common_chars = "的一是不了人在我有他这中到大来们个上为们地出以时天可下对生于也子得完蛋就"
# 扩充一些汉字
all_chars = list(set(common_chars + "甲乙丙丁戊己庚辛壬癸风云雨雪山川草木水火土石"))
char_to_idx = {ch: i+1 for i, ch in enumerate(all_chars)}  # 索引从1开始，0留作padding（本任务无用）
char_to_idx['你'] = 0  # 特殊字符“你”用0表示，方便定位
idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)
print(f"词汇表大小: {vocab_size}")

def generate_sample():
    """生成一个长度为5、包含且仅包含一个‘你’字的字符串，返回字符列表和‘你’的位置（0~4）"""
    pos = random.randint(0, 4)  # 你出现的位置
    chars = []
    for i in range(5):
        if i == pos:
            chars.append('你')
        else:
            chars.append(random.choice(all_chars))
    return chars, pos

def encode_sample(chars):
    """将字符列表转为索引列表"""
    return [char_to_idx[ch] for ch in chars]

# 生成数据集
num_samples = 5000
X_data = []   # 索引序列
y_data = []   # 位置标签（0~4）
for _ in range(num_samples):
    chars, pos = generate_sample()
    X_data.append(encode_sample(chars))
    y_data.append(pos)

# 划分训练集和验证集 (80% / 20%)
split = int(0.8 * num_samples)
train_X = torch.tensor(X_data[:split], dtype=torch.long)
train_y = torch.tensor(y_data[:split], dtype=torch.long)
val_X = torch.tensor(X_data[split:], dtype=torch.long)
val_y = torch.tensor(y_data[split:], dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=batch_size, shuffle=False)

# ----------------------------- 2. 模型定义 -----------------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        out, h_n = self.rnn(emb) # h_n: (1, batch, hidden_dim)
        # 取最后一层最后一个时间步的隐藏状态
        last_hidden = h_n[-1]    # (batch, hidden_dim)
        logits = self.fc(last_hidden)
        return logits

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)  # h_n: (1, batch, hidden_dim)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

# 超参数
embed_dim = 64
hidden_dim = 128
num_classes = 5
lr = 0.001
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化模型、损失函数、优化器
rnn_model = RNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
lstm_model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=lr)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=lr)

# ----------------------------- 3. 训练函数 -----------------------------
def train_model(model, optimizer, train_loader, val_loader, epochs, model_name):
    model.train()
    train_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, pred = torch.max(outputs, 1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        acc = correct / total
        val_accuracies.append(acc)
        print(f"[{model_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
        model.train()
    return train_losses, val_accuracies

# 训练 RNN 模型
print("\n----- 训练 RNN 模型 -----")
rnn_losses, rnn_accs = train_model(rnn_model, optimizer_rnn, train_loader, val_loader, epochs, "RNN")

# 训练 LSTM 模型
print("\n----- 训练 LSTM 模型 -----")
lstm_losses, lstm_accs = train_model(lstm_model, optimizer_lstm, train_loader, val_loader, epochs, "LSTM")

# ----------------------------- 4. 简单测试 ---------------------------------
def test_random_sample(model, char_to_idx, idx_to_char):
    chars, true_pos = generate_sample()
    idx_seq = torch.tensor([encode_sample(chars)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(idx_seq)
        pred = torch.argmax(logits, dim=1).item()
    print(f"文本: {''.join(chars)} | 真实位置: {true_pos+1} | 预测位置: {pred+1}")

print("\n----- 随机测试 RNN -----")
test_random_sample(rnn_model, char_to_idx, idx_to_char)
print("\n----- 随机测试 LSTM -----")
test_random_sample(lstm_model, char_to_idx, idx_to_char)
