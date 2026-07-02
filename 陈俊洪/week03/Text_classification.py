import os
import sys
import argparse
import numpy as np
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
# 数据标签
LABEL_NAMES = ['美食', '运动', '学习', '娱乐', '天气']

# objs对象词
# adjs形容词

# 超参数
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

CKPT_DIR    = os.path.dirname(__file__)
FIG_DIR     = os.path.join(os.path.dirname(__file__), 'figures')

random.seed(SEED)
torch.manual_seed(SEED)

CLASS_DATA = {
    0: {  # 美食
        'objs': ['火锅', '烧烤', '披萨', '寿司', '牛排', '拉面', '蛋糕', '奶茶', '小龙虾', '烤鱼'],
        'adjs': ['香', '辣', '鲜', '好吃', '美味', '地道', '酥脆', '浓郁'],
        'templates': [
            '今天中午吃了一顿{},味道真的很{}',
            '这家店的{}做得特别{},推荐大家来尝尝',
            '朋友请我吃{},口感{}到不行',
            '点了一份{}外卖,{}得让人满足',
            '第一次尝试{},没想到这么{}',
        ],
    },
    1: {  # 运动
        'objs': ['篮球', '足球', '羽毛球', '跑步', '游泳', '健身', '瑜伽', '骑行', '网球', '登山'],
        'adjs': ['激烈', '畅快', '累', '过瘾', '投入', '紧张', '放松'],
        'templates': [
            '下午和朋友一起去打{},打得非常{}',
            '坚持{}已经三个月了,感觉身体越来越{}',
            '周末报名参加了{}比赛,过程相当{}',
            '每天晚上都去{}半小时,整个人都{}下来了',
            '第一次尝试{},虽然{}但很有成就感',
        ],
    },
    2: {  # 学习
        'objs': ['数学', '英语', '编程', '物理', '机器学习', '算法', '历史', '语文', '化学'],
        'adjs': ['认真', '专注', '困难', '枯燥', '有趣', '吃力', '扎实'],
        'templates': [
            '今天复习了一整天的{},学得非常{}',
            '最近在自学{},感觉比想象中更{}',
            '准备{}考试,每天都{}地做题',
            '{}老师布置的作业需要{}才能完成',
            '这学期的{}课内容非常{},收获不少',
        ],
    },
    3: {  # 娱乐
        'objs': ['电影', '综艺', '电视剧', '演唱会', '游戏', '动漫', '剧本杀', '脱口秀'],
        'adjs': ['精彩', '搞笑', '感人', '紧张刺激', '无聊', '好看', '轻松'],
        'templates': [
            '昨晚熬夜看了一部{},剧情真的很{}',
            '周末和朋友一起玩{},过程相当{}',
            '最近追的那部{}越来越{}了',
            '去现场看了{},氛围特别{}',
            '推荐大家看看这个{},内容非常{}',
        ],
    },
    4: {  # 天气
        'objs': ['北京', '上海', '广州', '成都', '杭州', '这边', '我们这里', '老家'],
        'adjs': ['晴朗', '阴沉', '炎热', '凉爽', '潮湿', '干燥', '寒冷', '闷热'],
        'templates': [
            '今天{}的天气非常{},适合出门散步',
            '最近{}连续下了好几天雨,空气特别{}',
            '{}最近{}得不行,空调都不敢关',
            '早上起来看到{}外面{},心情都变好了',
            '{}的冬天{},记得多穿点衣服',
        ],
    },
}

class TextDataset(Dataset):
    """ 构建数据集 """
    def __init__(self, data, vocab):
        super().__init__()
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int=EMBED_DIM, hidden_dim: int=HIDDEN_DIM, dropout: int=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)
    
    def forward(self, x):
        e, _  = self.rnn(self.embedding(x))
        pooled = e.max(dim=1)[0] # 按照字的维度进行压缩,然后获取最大字的值和位置
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int=EMBED_DIM, hidden_dim: int=HIDDEN_DIM, dropout: int=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        e, _ = self.lstm(self.embedding(x))
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)


def make_sentence(label: int) -> str:
    info = CLASS_DATA[label]
    # 随机选择模板句子、对应的物品、以及形容词
    tmpl = random.choice(info['templates'])
    obj  = random.choice(info['objs'])
    adj  = random.choice(info['adjs'])
    try:
        return tmpl.format(obj, adj)
    except Exception:
        return obj + adj
    
def build_dataset(sample_num: int = N_SAMPLES):
    """ 为每一个标签创建2000个样本数据 """
    data = []

    # 获取标签号
    for label in CLASS_DATA.keys():
        # 对应标签下的所有样本数进行添加
        for _ in range(sample_num):
            data.append((make_sentence(label), label))
    random.shuffle(data)
    return data

def build_vocab(data):
    """ 为每一个字建立字表 """
    vocab = {'<PAD>':0, '<UNK>':1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, *, maxlen=MAXLEN):
    """ 把一句话转成定长的ID列表 
    :param sent: 输入的那句话
    :param vocab: 已经存在的字词表
    :return: 长度为 maxlen 的整数 ID 列表
    """
    
    ids = [vocab.get(ch, 1) for ch in sent] # 逐字查vocab里的ID,没有就是1
    ids = ids[:maxlen] # 截断句子的长度
    ids += [0] * (maxlen - len(ids)) # 将不够的赋0
    return ids

    
def make_bubbles():
    """ 处理生成数据集和字表 """
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    spilt = int(len(data) * TRAIN_RATIO)
    train_data = data[:spilt]
    test_data  = data[spilt:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TextDataset(test_data,  vocab), batch_size=BATCH_SIZE)

    return train_loader, test_loader, vocab

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred   = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total

def collect_predictions(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            ys.append(y)       # 返回真实标签
            ps.append(pred)    # 返回预测标签
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()      

def save_model(path, model, vocab):
    """只保存权重和偏置 """
    torch.save({
        'model_state': model.state_dict(),
        'vocab':       vocab,
    }, path)
    print(f"已保存模型权重 → {path}")

def load_model(path, model):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    print(f"已加载模型权重 ← {path}")
    return ckpt['vocab']

def train(train_loader, test_loader, model: nn.Module):
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_param = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_param:,}\n")

    history = {'train_loss': [], 'val_acc': []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss   = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, test_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率: {evaluate(model, test_loader):.4f}")
    return model, history

MODELS = {
    'rnn':  RNNModel,
    'lstm': LSTMModel,
}

def ckpt_path_for(model_name):
    return os.path.join(CKPT_DIR, f'{model_name}_ckpt.pth')

def plot_history(history, model_name, save_dir=FIG_DIR):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history['train_loss'], marker='o', color='tab:red')
    ax1.set_title(f'{model_name.upper()} — Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(epochs, history['val_acc'], marker='s', color='tab:blue')
    ax2.set_title(f'{model_name.upper()} — Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out = os.path.join(save_dir, f'{model_name}_loss_acc.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存曲线图 → {out}")

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir=FIG_DIR):
    os.makedirs(save_dir, exist_ok=True)
    class_names = ['Food', 'Sports', 'Study', 'Entertainment', 'Weather']
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f'{model_name.upper()} — Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存混淆矩阵 → {out}")

def tuili(model: nn.Module, vocab: dict):
    model.eval()
    test_sents = [
        '成都最近炎热得不行,空调都不敢关',
        '今天中午吃了一顿火锅,味道真的很香',
        '下午和朋友一起去打篮球,打得非常过瘾',
        '最近在自学机器学习,感觉比想象中更有趣',
        '昨晚熬夜看了一部电影,剧情真的很精彩',
    ]

    X = torch.tensor([encode(s, vocab) for s in test_sents], dtype=torch.long)
    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)

    print("\n推理结果:")
    for sent, pred, prob in zip(test_sents, preds, probs):
        label = LABEL_NAMES[pred.item()]
        conf  = prob[pred].item()
        print(f"  [{label}]  (置信度 {conf:.3f})  {sent}")

def parse_args():
    parser = argparse.ArgumentParser(
        description='文本分类：训练或加载已有模型做预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python Text_classification.py train --model rnn     # 训练 RNN\n'
            '  python Text_classification.py train --model lstm    # 训练 LSTM\n'
            '  python Text_classification.py train --model all     # 两个都训练\n'
            '  python Text_classification.py predict --model rnn   # 加载 RNN 权重预测\n'
            '  python Text_classification.py predict --model all   # 两个模型都预测\n'
        ),
    )
    parser.add_argument(
        'mode',
        choices=['train', 'predict'],
        help='train=从零训练并保存pth; predict=加载pth直接预测',
    )
    parser.add_argument(
        '--model', choices=['rnn', 'lstm', 'all'], default='rnn',
        help='选择模型: rnn / lstm / all,默认 rnn',
    )
    return parser.parse_args()

def run_train_one(model_name, train_loader, test_loader, vocab):
    print(f"\n{'='*20} 训练 {model_name.upper()} {'='*20}")
    ckpt_path = ckpt_path_for(model_name)
    model_cls = MODELS[model_name]

    model = model_cls(vocab_size=len(vocab))
    model, history = train(train_loader, test_loader, model)
    save_model(ckpt_path, model, vocab)
    plot_history(history, model_name)

    y_true, y_pred = collect_predictions(model, test_loader)
    plot_confusion_matrix(y_true, y_pred, model_name)
    tuili(model, vocab)

def run_predict_one(model_name):
    print(f"\n{'='*20} 预测 {model_name.upper()} {'='*20}")
    ckpt_path = ckpt_path_for(model_name)
    if not os.path.isfile(ckpt_path):
        print(f"[错误] 权重文件不存在: {ckpt_path}\n请先运行: python {sys.argv[0]} train --model {model_name}")
        return

    vocab = torch.load(ckpt_path, map_location='cpu')['vocab']
    model = MODELS[model_name](vocab_size=len(vocab))
    load_model(ckpt_path, model)

    data  = build_dataset(N_SAMPLES)
    spilt = int(len(data) * TRAIN_RATIO)
    test_loader = DataLoader(TextDataset(data[spilt:], vocab), batch_size=BATCH_SIZE)

    y_true, y_pred = collect_predictions(model, test_loader)
    plot_confusion_matrix(y_true, y_pred, model_name)
    tuili(model, vocab)

def main():
    args = parse_args()
    targets = ['rnn', 'lstm'] if args.model == 'all' else [args.model]

    if args.mode == 'train':
        train_loader, test_loader, vocab = make_bubbles()
        for name in targets:
            run_train_one(name, train_loader, test_loader, vocab)
    else:
        for name in targets:
            run_predict_one(name)

if __name__ == '__main__':
    main()
