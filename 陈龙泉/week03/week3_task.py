import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset,DataLoader

EMBED_DIM   = 32
HIDDEN_DIM  = 32
N_SAMPLES   = 4000
TRAIN_RATIO = 0.8
MAXLEN      = 5
BATCH_SIZE  = 64
LR          = 1e-3
EPOCHS      = 20




def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


#创建数据列表，其中表示要创建多少个样本
def build_dataset(n=N_SAMPLES):
    data = []
    words = "清晨的林间薄雾缓缓散开草木裹挟着湿润的清气漫溢开来脚下的泥土松软温润散落的枯叶层层叠叠踩上去发出细碎轻柔的声响枝干交错伸展嫩绿的新叶缀满枝头微风拂过枝叶轻轻摇曳光影在地面错落晃动远处的溪流顺着青石蜿蜒流淌水声潺潺清冽的水流绕过碎石泛起细碎的涟漪岸边丛生的野草肆意生长各色细碎野花悄然绽放淡白浅黄浅紫点缀在青绿之间朴素却自有生机林间偶尔传来飞鸟轻鸣清越婉转打破静谧又融于静谧阳光穿过枝叶缝隙洒落化作斑驳光点落在草丛与石径之上远离城市喧嚣周遭只剩自然的平和心绪慢慢沉静浮躁渐渐消散只剩松弛与安然万物遵循时序缓缓生长不急不躁山野之间每一寸草木每一缕清风都藏着平淡又治愈的力量简单纯粹安稳绵长"
    list1 = list(words)
    for _ in range(N_SAMPLES):
        sample = random.sample(list1,4)
        x = random.randint(0,4)
        sample.insert(x,"你")
        data.append((sample,sample.index("你")))
    random.shuffle(data)
    #print(data)
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab



class KeyWordSplit(nn.Module):
    """
        对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        #将中文转换成词表，其中vocab_size表示输入的几个汉子，embed_dim表示每个字是几维向量，padding_idx=0表示全0的字符向量不更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        #循环层，主要是用于语序的分析，其中embed_dim表示输入的向量维度，hidden_dim表示输出的向量维度
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        #归一化层，对于每句话的字进行归一化，这样更容易训练
        self.ln        = nn.LayerNorm(hidden_dim)
        #随机抛弃部分特征，防止过拟合
        self.dropout   = nn.Dropout(dropout)
        #全连接层，输入
        self.fc        = nn.Linear(hidden_dim, 5)
    
    def forward(self, x):
        # x: (batch, seq_len)
        # 输入: (一批样本几句话，每个样本几个字)
        out = self.embedding.forward(x)
        # 输入: (一批样本几句话，每个样本几个字，每个字的向量维度)
        e, _ = self.rnn.forward(out)  # (B, L, hidden_dim)
        # 池化层
        # 输出:((一批样本几句话，每句话的向量维度))，其中[0]表示只要最大值的值，[1]表示最大值的位置
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        # 输出:((一批样本几句话，每句话的向量维度(归一化之后每个向量都缩小了)))
        pooled = self.ln.forward(pooled)
        # 输出:((一批样本几句话，每句话的向量维度(归一化之后每个向量都缩小了，随机抛弃了一部分向量)))
        pooled = self.dropout.forward(pooled)
        # 输出:((一批样本几句话，每句话用一个5维向量表示)))
        pooled = self.fc.forward(pooled)
        # 输出:((一批样本几句话，每句话用一个5维向量表示，且5维向量加和为1))),对最后一个维度做归一化
        out = torch.softmax(pooled,dim=-1)  # (B,)
        return out

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob    = model(X)
            pred    = torch.argmax(prob,dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total

def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")
    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeyWordSplit(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")


if __name__ == '__main__':
    train()
