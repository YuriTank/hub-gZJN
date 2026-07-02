'''【第三周作业内容：】
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。如果不知道怎么设计，
可以选择如下任务:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader

#  自定义生成文本，并完成数据分类
def generate_data(num):
    base_chars = ['好', '中', '国', '人', '坏', '聪', '明', '笨', '蛋', '懒', '汉']
    X = []
    Y = []
    for _ in range(num):
        # 严格生成 4 个随机字符
        chars = random.sample(base_chars, 4)
        # 在 0 到 4 之间随机选一个位置插入"你"
        pos = random.randint(0, 4)
        chars.insert(pos, '你')
        text = ''.join(chars)  
        X.append(text)
        Y.append(pos)        
    return X, Y

#  构建词表 
def build_vocab(texts):
    chars = set(''.join(texts))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ch in sorted(chars):
        vocab[ch] = len(vocab)
    return vocab

# 编码并输出索
def encode(sent, vocab, maxlen):
    ids = [vocab.get(c, 1) for c in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# 构建数据源
def text_data(num, maxlen, existing_vocab=None):
    X, Y = generate_data(num)
    # 如果传入了已有词表（测试时），就用传入的；否则新建（训练时）
    vocab = existing_vocab if existing_vocab else build_vocab(X)
    vocab_size = len(vocab)
    
    x_encode = [encode(text, vocab, maxlen) for text in X]
    x_tensor = torch.tensor(x_encode, dtype=torch.long)
    y_tensor = torch.tensor(Y, dtype=torch.long) 
    return x_tensor, y_tensor, vocab_size, vocab

# 构建RNN模型 
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        pooled  = output.max(dim=1)[0] 
        pooled = self.dropout(pooled) 
        return self.fc(pooled)

# 训练与评估
criterion = nn.CrossEntropyLoss()

def evaluate(model, vocab):
    model.eval()
    test_num = 100
    # 传入训练时的 vocab！
    x, y, _, _ = text_data(test_num, maxlen=5, existing_vocab=vocab)
    
    print(f'预测集样本分布: {np.unique(y.numpy(), return_counts=True)}')
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        predict_classes = torch.argmax(y_pred, dim=1) 
        for y_p, y_t in zip(predict_classes, y):  
            if y_p == y_t: correct += 1  
            else: wrong += 1 
    acc = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f\n" % (correct, acc))
    return acc


# 训练模型
def train():
    # 超参数
    num = 1000
    maxlen = 5
    embedding_dim = 64   
    hidden_dim = 16
    output_dim = 5
    learning_rate = 0.01 
    epoch_num = 50
    batch_size = 32  
    # 构建数据
    x, y, vocab_size, vocab = text_data(num, maxlen)
    # 将 x 和 y 包装成 Dataset
    dataset = TensorDataset(x, y)
    # 用 DataLoader 将 Dataset 变成可迭代的批次生成器  
    # shuffle=True 会在每个 epoch 自动打乱数据！
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TextRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)  
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        # 每轮结束后，用测试集评估模型
        with torch.no_grad():
            total_loss = criterion(model(x), y)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch_num, total_loss.item()))
     # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 测试本轮模型结果
    acc = evaluate(model, vocab)  
    print('Accuracy: {:.4f}\n'.format(acc))    
if __name__ == '__main__':
    train()
