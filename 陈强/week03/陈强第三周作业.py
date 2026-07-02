import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备一个极小的人工数据集（5个样本，每个样本是一个5字文本，标签是“你”的位置-1）
# 字符到数字的映射（共4个字符：你、我、他、好）
char_to_idx = {'你': 0, '我': 1, '他': 2, '好': 3}

# 训练数据：每个样本是一个整数列表，标签是类别（0~4）
train_texts = [
    [0, 1, 2, 3, 1],   # "你 我 他 好 我" -> 标签 0（'你'在第1位）
    [1, 0, 3, 2, 1],   # "我 你 好 他 我" -> 标签 1（'你'在第2位）
    [1, 2, 0, 3, 1],   # "我 他 你 好 我" -> 标签 2
    [3, 1, 2, 0, 3],   # "好 我 他 你 好" -> 标签 3
    [2, 3, 1, 2, 0]    # "他 好 我 他 你" -> 标签 4
]
train_labels = [0, 1, 2, 3, 4]   # 类别对应位置

# 验证集
val_texts = [[1, 0, 1, 2, 3]]   # "我 你 我 他 好" -> 标签应为1
val_labels = [1]

# 超参数
vocab_size = len(char_to_idx)   # 4
embed_size = 4                  # 嵌入维度
hidden_size = 8                 # RNN隐藏层大小
num_classes = 5                 # 5个类别（位置1~5）
learning_rate = 0.01
epochs = 200

# 2. 定义模型（使用RNN）
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x形状: (batch_size, seq_len) -> 这里 batch_size=1
        embedded = self.embedding(x)          # (1, seq_len, embed_size)
        out, hidden = self.rnn(embedded)      # out: (1, seq_len, hidden_size)
        # 取最后一个时间步的输出（也可以取hidden[-1]）
        last_output = out[:, -1, :]           # (1, hidden_size)
        logits = self.fc(last_output)         # (1, num_classes)
        return logits

model = SimpleRNNClassifier(vocab_size, embed_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 3. 训练循环
for epoch in range(epochs):
    total_loss = 0
    # 每个epoch遍历所有训练样本
    for i, (text, label) in enumerate(zip(train_texts, train_labels)):
        # 转换为tensor，形状 (1, seq_len)
        x = torch.tensor([text], dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)

        # 前向传播
        outputs = model(x)          # (1, 5)
        loss = criterion(outputs, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 每20轮输出一次损失，并在验证集上测试
    if (epoch+1) % 20 == 0:
        # 验证
        with torch.no_grad():
            val_x = torch.tensor(val_texts, dtype=torch.long)
            val_y = torch.tensor(val_labels, dtype=torch.long)
            val_out = model(val_x)
            _, predicted = torch.max(val_out, 1)
            acc = (predicted == val_y).float().mean().item()
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_texts):.4f} | Val Acc: {acc:.2f}")

# 4. 测试
def predict(text_str):
    # 将字符串转为索引列表，长度必须为5
    idx_list = [char_to_idx[ch] for ch in text_str]
    x = torch.tensor([idx_list], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        pred_class = torch.argmax(logits, dim=1).item() + 1   # +1变成1~5的位置
    return pred_class

test_str = "我你好他好"   # 实际索引 [1,0,2,3,2] -> '你'在第2位，应输出2
print(f"输入文本: {test_str} -> 预测'你'在第 {predict(test_str)} 位")
