"""
文本多分类任务实验 - RNN/LSTM/GRU对比
任务：对任意包含"你"字的五字文本，根据"你"字的位置进行分类（0-4共5类）

author: zxm
Date: 2026/4/30
"""
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 5000  # 增加样本量
SEQ_LEN = 5       # 固定为5个字
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 30
TRAIN_RATIO = 0.8
MODEL_TYPES = ['RNN', 'LSTM', 'GRU']  # 要实验的模型类型

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
def generate_five_char_text():
    """生成一个五字文本，其中包含一个'你'字"""
    # 常用汉字集合（简化版）
    common_chars = '的一是不了人在有我他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三己老从动两长把身但样与想现美'
    
    # 随机生成4个其他汉字
    other_chars = ''.join(random.choices(common_chars, k=4))
    
    # 随机选择"你"字的位置（0-4）
    pos = random.randint(0, 4)
    
    # 构造五字文本
    text = other_chars[:pos] + '你' + other_chars[pos:]
    
    return text, pos

def build_dataset(n=N_SAMPLES):
    """构建数据集"""
    data = []
    for _ in range(n):
        text, label = generate_five_char_text()
        data.append((text, label))
    
    # 检查类别分布
    label_dist = Counter([label for _, label in data])
    print("类别分布:", dict(sorted(label_dist.items())))
    
    random.shuffle(data)
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    """构建词表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(text, vocab, seq_len=SEQ_LEN):
    """将文本编码为索引序列"""
    ids = [vocab.get(ch, 1) for ch in text]  # 1是<UNK>的索引
    ids = ids[:seq_len]
    ids += [0] * (seq_len - len(ids))  # 用0填充
    return ids

# ─── 3. Dataset / DataLoader ────────────────────────────────
class FiveCharDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(text, vocab) for text, _ in data]
        self.y = [label for _, label in data]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class CharPositionClassifier(nn.Module):
    """字符位置分类器基类"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=5, model_type='RNN'):
        super().__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 根据模型类型选择不同的RNN单元
        if model_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # RNN处理
        if self.model_type == 'LSTM':
            rnn_out, (hidden, _) = self.rnn(embedded)
        else:
            rnn_out, hidden = self.rnn(embedded)
        
        # 取最后一个时间步的隐藏状态
        if self.model_type == 'LSTM':
            last_hidden = hidden[-1]  # LSTM返回的是(h_n, c_n)
        else:
            last_hidden = hidden[-1]  # RNN/GRU返回的是h_n
        
        # 全连接层分类
        out = self.fc(self.dropout(last_hidden))
        return out

# ─── 5. 训练与评估函数 ──────────────────────────────────────
def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    """训练单个模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print(f"\n{'='*50}")
    print(f"开始训练 {model_name} 模型")
    print(f"{'='*50}")
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)
        
        # 学习率调整
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        
        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"{model_name} 最佳验证准确率: {best_val_acc:.4f}")
    return train_losses, val_accuracies, best_val_acc

def evaluate_model(model, test_loader, model_name):
    """评估模型在测试集上的表现"""
    model.eval()
    correct = 0
    total = 0
    confusion_matrix = np.zeros((5, 5), dtype=int)
    
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # 构建混淆矩阵
            for i in range(len(y)):
                confusion_matrix[y[i].item()][predicted[i].item()] += 1
    
    accuracy = correct / total
    
    print(f"\n{model_name} 测试结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  正确数/总数: {correct}/{total}")
    
    # 打印每个类别的准确率
    print("\n  各类别准确率:")
    for i in range(5):
        class_correct = confusion_matrix[i][i]
        class_total = np.sum(confusion_matrix[i])
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"    类别{i}(位置{i}): {class_acc:.4f} ({class_correct}/{class_total})")
    
    return accuracy, confusion_matrix

def test_examples(model, vocab, examples):
    """测试具体例子"""
    model.eval()
    print("\n测试例子:")
    print("-" * 40)
    
    with torch.no_grad():
        for text in examples:
            # 编码
            ids = encode(text, vocab)
            input_tensor = torch.tensor([ids], dtype=torch.long)
            
            # 预测
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            
            # 实际位置
            actual_pos = text.find('你')
            
            print(f"文本: '{text}'")
            print(f"  预测位置: {pred_class} (置信度: {confidence:.4f})")
            print(f"  实际位置: {actual_pos}")
            print(f"  {'✓' if pred_class == actual_pos else '✗'}")
            print("-" * 40)

# ─── 6. 主实验函数 ──────────────────────────────────────────
def main_experiment():
    print("="*60)
    print("文本多分类任务实验 - RNN/LSTM/GRU对比")
    print("任务: 预测五字文本中'你'字的位置(0-4)")
    print("="*60)
    
    # 1. 准备数据
    print("\n1. 生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"   样本总数: {len(data)}")
    print(f"   词表大小: {len(vocab)}")
    print(f"   类别数: 5 (位置0-4)")
    
    # 划分数据集
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 创建DataLoader
    train_dataset = FiveCharDataset(train_data, vocab)
    val_dataset = FiveCharDataset(val_data, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 2. 实验不同模型
    results = {}
    
    for model_type in MODEL_TYPES:
        print(f"\n2. 创建{model_type}模型...")
        model = CharPositionClassifier(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=5,
            model_type=model_type
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   {model_type}总参数量: {total_params:,}")
        print(f"   可训练参数量: {trainable_params:,}")
        
        # 训练模型
        train_losses, val_accuracies, best_val_acc = train_model(
            model, train_loader, val_loader, model_type
        )
        
        # 保存结果
        results[model_type] = {
            'model': model,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    # 3. 结果对比
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    
    # 创建测试集
    test_data = build_dataset(1000)  # 额外的1000个样本用于测试
    test_dataset = FiveCharDataset(test_data, vocab)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    test_results = {}
    for model_type in MODEL_TYPES:
        # 加载最佳模型
        model = results[model_type]['model']
        model.load_state_dict(torch.load(f'best_{model_type}.pth'))
        
        # 在测试集上评估
        test_acc, confusion_matrix = evaluate_model(model, test_loader, model_type)
        test_results[model_type] = {
            'test_accuracy': test_acc,
            'confusion_matrix': confusion_matrix
        }
    
    # 4. 可视化训练过程
    plt.figure(figsize=(12, 4))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    for model_type in MODEL_TYPES:
        plt.plot(results[model_type]['train_losses'], label=f'{model_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 验证准确率曲线
    plt.subplot(1, 2, 2)
    for model_type in MODEL_TYPES:
        plt.plot(results[model_type]['val_accuracies'], label=f'{model_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("\n训练曲线已保存为 'training_comparison.png'")
    
    # 5. 测试具体例子
    test_examples_list = [
        '你今天好吗',    # 位置0
        '问你好不好',    # 位置1
        '谢谢你的爱',    # 位置2
        '我爱你的心',    # 位置3
        '永远爱着你',    # 位置4
        '你你你你你',    # 多个"你"字（第一个位置）
        '没有你字啊',    # 没有"你"字（应该预测为<UNK>）
    ]
    
    # 用LSTM模型测试（通常表现最好）
    lstm_model = results['LSTM']['model']
    lstm_model.load_state_dict(torch.load('best_LSTM.pth'))
    test_examples(lstm_model, vocab, test_examples_list)
    
    # 6. 总结报告
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    
    print("\n各模型最终测试准确率:")
    for model_type in MODEL_TYPES:
        test_acc = test_results[model_type]['test_accuracy']
        best_val_acc = results[model_type]['best_val_acc']
        print(f"  {model_type}:")
        print(f"    最佳验证准确率: {best_val_acc:.4f}")
        print(f"    测试准确率: {test_acc:.4f}")
    
    print("\n分析:")
    print("1. LSTM通常表现最好，因为能更好地捕捉长期依赖")
    print("2. GRU参数量较少，训练更快，但性能接近LSTM")
    print("3. 简单RNN可能因梯度消失/爆炸问题表现稍差")
    print("4. 所有模型都应能学到'你'字的位置模式")
    
    print("\n扩展建议:")
    print("1. 尝试增加文本长度（如10字、20字）")
    print("2. 实验双向RNN/LSTM/GRU")
    print("3. 添加注意力机制")
    print("4. 使用预训练词向量")
    print("5. 增加更多汉字和更复杂的句子模式")

# ─── 主程序 ────────────────────────────────────────────────
if __name__ == '__main__':
    main_experiment()
