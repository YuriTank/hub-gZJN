import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
N_SAMPLES = 5000
MAXLEN = 5
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5

random.seed(SEED)
torch.manual_seed(SEED)

COMMON_CHARS = ['我', '他', '她', '它', '们', '的', '是', '在', '有', '好',
                '今', '天', '明', '昨', '早', '晚', '上', '下', '中', '大',
                '小', '多', '少', '高', '低', '快', '慢', '真', '很', '太',
                '吗', '呢', '啊', '呀', '吧', '哦', '嗯', '了', '过', '来',
                '去', '回', '看', '听', '说', '做', '吃', '喝', '玩', '乐']


def generate_sentence(position):
    chars = [random.choice(COMMON_CHARS) for _ in range(5)]
    chars[position - 1] = '你'
    return ''.join(chars)


def build_dataset(n=N_SAMPLES):
    data = []
    samples_per_class = n // NUM_CLASSES

    for pos in range(1, NUM_CLASSES + 1):
        for _ in range(samples_per_class):
            sent = generate_sentence(pos)
            label = pos - 1
            data.append((sent, label))

    random.shuffle(data)
    return data


def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


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


class PositionRNN(nn.Module):

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e, _ = self.rnn(
            self.embedding(x))
        last_output = e[:, -1, :]
        out = self.fc(last_output)
        return torch.softmax(out, dim=1)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            probs = model(X)
            pred = probs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train():
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            probs = model(X)

            loss = criterion(torch.log(probs + 1e-8), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(
            f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '今天你好吗',
        '我真棒啊你',
        '我们你好呀',
        '大家好你哦',
        '今天见到你',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            probs = model(ids)
            prob_list = probs[0].tolist()
            formatted_probs = [f"{p:.4f}" for p in prob_list]
            print(
                f"'{sent}': [{', '.join(formatted_probs)}]")


if __name__ == '__main__':
    train()
