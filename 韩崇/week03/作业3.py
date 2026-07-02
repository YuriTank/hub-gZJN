
#设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
# 如果不知道怎么设计，可以选择如下任务:
# 对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


CHARS = ['我', '他', '她', '们', '爱', '学', '习', '深', '度', '好', '吗', '呀', '很', '棒', '赞', '欢', '满', '的', '了', '在', '是', '有', '和', '不', '就', '也', '还', '要', '去', '来', '看', '吃', '喝', '玩', '学', '习', '编', '程']
TARGET_CHAR = '你'


# ─── 1. 数据生成 ────────────────────────────────────────────
def make_sample():
    pos = random.randint(0, 4)
    chars = random.choices(CHARS, k=5)
    chars[pos] = TARGET_CHAR
    return ''.join(chars), pos

def build_dataset(n=1000):
    data = []
    for _ in range(n):
        data.append(make_sample())
    random.shuffle(data)
    return data


def show_dataset_info(data, vocab, title='数据集信息', n=10):
    print(f'--- {title} ---')
    print(f'样本数量: {len(data)}')
    print(f'词表大小: {len(vocab)}')

    class_counts = [0] * 5
    for _, label in data:
        class_counts[label] += 1
    for i, count in enumerate(class_counts):
        print(f'第{i + 1}位是"{TARGET_CHAR}"的样本数: {count}')

    print('\n--- 数据集示例 ---')
    for sent, label in data[:n]:
        print(f'文本: {sent}  标签: {label}  含义: "{TARGET_CHAR}"在第{label + 1}位')


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab):
    return [vocab.get(ch, 1) for ch in sent]


def evaluate_data(model, data, vocab, title='测试集', n=10):
    model.eval()
    correct = 0
    examples = []

    with torch.no_grad():
        for sent, true_label in data:
            encoded = encode(sent, vocab)
            ids = torch.tensor([encoded], dtype=torch.long)
            outputs = model(ids)
            pred_label = torch.argmax(outputs, dim=1).item()

            if pred_label == true_label:
                correct += 1

            if len(examples) < n:
                examples.append((sent, encoded, true_label, pred_label))

    print(f'\n--- {title}准确率 ---')
    print(f'Accuracy: {correct / len(data) * 100:.2f}%')

    print(f'\n--- {title}预测示例 ---')
    for sent, ids, true_label, pred_label in examples:
        result = '正确' if true_label == pred_label else '错误'
        print(
            f'文本: {sent}  '
            f'编码: {ids}  '
            f'真实: 第{true_label + 1}位  '
            f'预测: 第{pred_label + 1}位  '
            f'判断: {result}'
        )


class NiPositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(sent, vocab) for sent, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long)
        )
    


# 定义模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 定义超参数
batch_size = 16
num_samples = 1000
train_ratio = 0.8

data = build_dataset(num_samples)
split = int(len(data) * train_ratio)
train_data = data[:split]
test_data = data[split:]

vocab = build_vocab(train_data)
show_dataset_info(train_data, vocab, title='训练集信息')
show_dataset_info(test_data, vocab, title='测试集信息')
train_dataset = NiPositionDataset(train_data, vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

embed_dim = 10
hidden_size = 20
output_size = 5
num_epochs = 100
learning_rate = 0.001

model = RNNClassifier(len(vocab), embed_dim, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# 用没参与训练的数据测试
evaluate_data(model, test_data, vocab, title='未见过的测试集')

# 这些句子包含训练词表中没有的字，编码时会变成 <UNK> 的编号 1
oov_test_data = [
    ('新你闻报纸', 1),
    ('电灯你光路', 2),
    ('饭菜很你香', 3),
    ('雨伞车站你', 4),
]
evaluate_data(model, oov_test_data, vocab, title='含词表外字符的测试集', n=4)
