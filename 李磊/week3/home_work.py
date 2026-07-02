"""
输入一个文本判断你所在的索引
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import onnx
import onnxsim
import numpy as np

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 20
N_SAMPLES   = 8000  # 样本数量
MAXLEN      = 32    # 句子最大长度
EMBED_DIM   = 64    # 词向量维度
HIDDEN_DIM  = 64    # RNN 隐藏层维度
LR          = 1e-3  # 学习率
BATCH_SIZE  = 64    # 批大小
EPOCHS      = 40   # 训练轮数
TRAIN_RATIO = 0.8   # 训练集比例

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
POS_KEYS = ['你']
RANDOM_CHARS = '的一是在不了有人和这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处理世车身书布'
MIN_RANDOM_LEN = 20
MAX_RANDOM_LEN = 31
MIN_KEY_POS = 0
MAX_KEY_POS = 31


def make_random_text(min_len=MIN_RANDOM_LEN, max_len=MAX_RANDOM_LEN):
    length = random.randint(min_len, max_len)
    return ''.join(random.choice(RANDOM_CHARS) for _ in range(length))


def insert_keyword_into_random_text(keyword, prefix_len=None):
    if prefix_len is None:
        prefix_len = random.randint(MIN_KEY_POS, MAX_KEY_POS)
    prefix = make_random_text(prefix_len, prefix_len)
    suffix = make_random_text()
    sent = prefix + keyword + suffix
    label = len(prefix)
    return sent, label

def make_positive():
    kw = random.choice(POS_KEYS)
    return insert_keyword_into_random_text(kw)


def make_negative():
    sent = make_random_text()
    return sent, MAXLEN


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(make_positive())
    random.shuffle(data)
    return data

# ─── 2. Dataset / DataLoader ────────────────────────────────
# 词表映射构建:w2index
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

# 文本转索引向量:ws2indexs,并向MAXLEN对齐
def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

#  PyTorch数据集类，配合DataLoader按batch划分数据集
# __len__和__getitem__必须实现
class TextDataset(Dataset):
    # 初始化：将原始文本转换为模型输入向量
    # data:数据+标签
    # vocab:词表
    def __init__(self, data, vocab):
        # 文本转向量
        self.X = []
        for item in data:
            s, _ = item                 # 文本
            encoded = encode(s, vocab)  # 文本转索引向量，向MAXLEN对齐
            self.X.append(encoded)
        # 标签
        self.y = [lb for _, lb in data]

    # 获取数据集大小
    def __len__(self):
        return len(self.y)

    # 按索引获取数据
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 3. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_dim * 2, 1)
        self.softmax    = nn.Softmax(dim=1)
        self.loss      = torch.nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        emb = self.embedding(x)             # 文字索引转向量：(B, MAXLEN) --> (B, MAXLEN, EMBED_DIM)
        e, _ = self.lstm(emb)               # RNN：(B, MAXLEN, EMBED_DIM) --> (B, MAXLEN, 2*HIDDEN_DIM)
        drop_out = self.dropout(e)          # Dropout：(B, MAXLEN, 2*HIDDEN_DIM) --> (B, MAXLEN, 2*HIDDEN_DIM)
        fc_out = self.fc1(drop_out)         # 逐位置打分：(B, MAXLEN, 2*HIDDEN_DIM) --> (B, MAXLEN, 1)
        logits = fc_out.squeeze(2)          # 去除多余维度：(B, MAXLEN, 1) --> (B, MAXLEN)
        if label is not None:               # 计算loss
            loss = self.loss(logits, label)
            return loss
        logits = self.softmax(logits)        # 概率化：(B, MAXLEN) --> (B, MAXLEN)
        return logits
    
    def getLoss(self, x, label):
        return self.forward(x, label)

# ─── 4. 导出ONNX模型 ──────────────────────────────────────────
def export_onnx(model, filepath='model/keyword_rnn.onnx'):
    model.eval()
    dummy_input = torch.zeros(1, MAXLEN, dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=17,
        dynamo=False,
    )
    
    model_onnx = onnx.load(filepath)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, filepath)
    print(f"ONNX模型已导出至: {filepath}")


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    total_distance = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y.long()).sum().item()
            total += y.size(0)
            total_distance += torch.abs(pred - y.long()).sum().item()
    acc = correct / total
    avg_distance = total_distance / total
    return acc, total_distance, avg_distance


def train():
    """数据集生成"""
    data  = build_dataset(N_SAMPLES)    # 生成数据集
    vocab = build_vocab(data)           # 生成词表
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    # 划分训练集和验证集
    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]   # 训练集
    val_data   = data[split:]   # 验证集

    # 按batch_size划分数据集
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    """创建模型、损失函数和优化器""" 
    model     = KeywordRNN(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    """训练"""
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            # 计算loss
            loss = model.getLoss(X, y)

            # 反向传播：计算梯度并更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, val_distance, val_avg_distance = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_distance={val_distance:.0f}  val_avg_distance={val_avg_distance:.4f}")

    val_acc, val_distance, val_avg_distance = evaluate(model, val_loader)
    print(f"\n最终验证准确率：{val_acc:.4f}，总位置误差：{val_distance:.0f}，平均位置误差：{val_avg_distance:.4f}")

    """推理示例"""
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [insert_keyword_into_random_text('你', pos) for pos in range(32)]
    with torch.no_grad():
        for sent, target_pos in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            pred_idx = torch.argmax(model(ids), dim=1).item()
            distance = abs(pred_idx - target_pos)
            is_correct = pred_idx == target_pos
            print(f"  [真实位置={target_pos}, 预测位置={pred_idx}, 是否准确={is_correct}, 索引距离={distance}, 文本长度={len(sent)}]  {sent}")

    export_onnx(model)  # 导出ONNX模型


if __name__ == '__main__':
    train()