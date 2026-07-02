import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random


'''设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
如果不知道怎么设计，可以选择如下任务:对一个任意包含“你”字的五个字的文本，
“你”在第几位，就属于第几类。
'''


# 1 定义超参数
SEED        = 42
N_SAMPLES   = 3000 # 样本数量
MAX_LEN     = 5    # 句子最大长度
EMBED_DIM   = 64   # 词向量维度
HIDDEN_DIM  = 64   # 隐藏层维度
LR          = 1e-3 # 学习率
BATCH_SIZE  = 64   # 批量大小
EPOCHS      = 20   # 迭代轮数
TRAIN_RATIO = 0.8  # 训练集比例


# 设置随机种子以保证结果可复现
random.seed(SEED)
torch.manual_seed(SEED)

BASE_CHARS = "是的撒多次尴尬女卡电话阿萨德二哥sdkjadjaccf你"


# 2. 随机生成数据与预处理
def generate_data(n_samples=20):
    NI_COUNT = 5
    COMMON_CHARS = BASE_CHARS + "你" * NI_COUNT
    data = []
    for _ in range(n_samples):
        chars = random.choices(COMMON_CHARS, k=MAX_LEN)
        sentence = "".join(chars)
        if '你' in sentence:
            label = sentence.index('你')
        else:
            label = -1
        data.append((sentence, label))

    return data

# 构建词表
def build_vocab(data):
    vocab = {'<PAD>':0, 'UNK':1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

# 填充
def encode_sentence(sentence, vocab, max_len=MAX_LEN):
    ids  = [vocab.get(ch, 1) for ch in sentence]
    ids  = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# 构建数据集
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = [encode_sentence(s, vocab) for s, _ in data]
        self.y = []
        for _, lb in data:
            # 不包含你的分为5类
            if lb == -1:
                self.y.append(5)
            else:
                self.y.append(lb)

    def __len__(self):
        return len(self.y)
    
    # 返回一个样本
    def __getitem__(self, index):
        return (
            torch.tensor(self.x[index], dtype=torch.long),
            torch.tensor(self.y[index], dtype=torch.long)
        )

# 定义RNN模型
class keywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes = 6, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)
   
    def forward(self, x):
        # 
        embed = self.embedding(x)
        rnn_out, hidden = self.rnn(embed)
        last_hidden = hidden[-1]
        out = self.dropout(self.bn(last_hidden))
        logits = self.fc(out)
        return logits


# LSTM
class keywordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes = 6, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)
   
    def forward(self, x):
        # 
        embed = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embed)
        last_hidden = hidden[-1]
        out = self.dropout(self.bn(last_hidden))
        logits = self.fc(out)
        return logits


# 训练 评估
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    # 
    with torch.no_grad():
        for x, y in loader:
            prob = model(x)
            _, pred = torch.max(prob, dim=1) # 
            correct += (pred == y).sum().item()
            total += y.size(0)

        return correct / total

# 模型训练
def train():
    data = generate_data(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本数：{len(data)}, 词表大小:{len(vocab)}")
    # 划分数据集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    # 创建数据加载器
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # RNN
    # model = keywordRNN(vocab_size=len(vocab))
    # LSTM
    model = keywordLSTM(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量:{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    model.eval()
    
    test_sents = [
        '你好，中国',
        '今天这么好',
        '明天你去找',
        'ssss你',
        '好你你你你',
    ]

    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode_sentence(sent, vocab)], dtype=torch.long)
            prob  = torch.softmax(model(ids), dim=1)
            _, pred = torch.max(prob, dim=1)
            
            print(f'样本的分类：{pred}, 概率为：{prob}')
            
    
if __name__ == '__main__':
    train()
