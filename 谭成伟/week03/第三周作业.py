import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
import json


class TorchRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        # 无需使用Python 2写法 super(TorchRNN, self).__init__()
        # Python 3 简化写法,
        super().__init__()
        # 添加embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer = nn.RNN(embedding_dim, hidden_size, bias=False, batch_first = True)
        # 添加线性层，将hidden_size映射到词表大小
        self.fc = nn.Linear(hidden_size, vocab_size)
        # 使用交叉熵损失，自带softmax操作
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """前向计算"""
        # 经过embedding层处理
        x_emb = self.embedding(x)
        # 经过RNN模型计算后的结果
        output, hidden = self.layer(x_emb)
        # 通过线性层映射到词表大小
        y_pred = self.fc(output)
        if y is not None:
            # 将y从one-hot编码转换为下标
            # y_indices = torch.argmax(y, dim=-1)
            loss = self.loss(y_pred, y)
            # 返回损失值
            return loss
        else:
            # 返回模型预测的结果
            return torch.softmax(y_pred, dim=-1)


def build_vocab():
    """
    构建词表
    :return:
    """
    chars = "五一劳动节快乐"
    vocab = {}
    for i, char in enumerate(chars):
        vocab[char] = i
    # print(vocab)
    return vocab

def build_sample(vocab, total_sample_num, str_size):
    """
    构建样本
    :param vocab: 词表
    :param total_sample_num: 共构建多少批数据
    :return: 样本，输入x，输出y
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        keys = list(vocab.keys())
        sample = random.sample(keys, str_size)
        # 将字符转换为下标
        x = [vocab[char] for char in sample]
        X.append(x)
        # 1的位置代表当前字符下标的位置
        sample_y = []
        vocab_size = len(vocab)
        for char in sample:
            one_hot = [0] * vocab_size
            one_hot[vocab[char]] = 1
            sample_y.append(one_hot)
        Y.append(sample_y)
    return torch.LongTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    str_size = 5
    vocab = build_vocab()
    x, y = build_sample(vocab, test_sample_num, str_size)
    print(f"样本数量：{y.size()}")
    print(f"本次预测集中共有{y.size()}个样本" )
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        y_pred_indices = torch.argmax(y_pred, dim=-1)
        # 将y从one-hot编码转换为下标
        y_indices = torch.argmax(y, dim=-1)
        for y_p, y_t in zip(y_pred_indices, y_indices):
            if torch.all(y_p == y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def predict(model_path, vocab_path, input_strings):
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    vocab_size = len(vocab)
    embedding_dim = 12
    hidden_size = 5
    model = TorchRNN(vocab_size, embedding_dim, hidden_size)
    model.load_state_dict(torch.load(model_path))
    for input_string in input_strings:
        # 将字符串拆分为单个字符
        chars = list(input_string)
        # 将每个字符转换为下标
        x = [vocab[char] for char in chars]
        # 添加批次维度
        x_tensor = torch.LongTensor([x])
        model.eval()
        with torch.no_grad():
            y_pred = model(x_tensor)
        # 打印每个字符的预测结果
        print(f"输入字符串：{input_string}")
        for i, char in enumerate(chars):
            pred_idx = torch.argmax(y_pred[0][i]).item()
            pred_prob = y_pred[0][i][pred_idx].item()
            print(f"  字符：{char}, 预测位置：{pred_idx}, 概率值：{pred_prob:.4f}")

def main():
    learning_rate = 0.05  # 学习率降低
    vocab = build_vocab()
    vocab_size = len(vocab)
    embedding_dim = 12  # 每个字符的embedding维度
    hidden_size = 5  # 增加hidden_size提高模型容量
    batch_size = 20 # 每次训练样本个数
    train_sample = 500 # 增加每轮训练样本总数
    epoch_num = 50 # 训练轮数
    str_size = 5 # 字符长度
    model = TorchRNN(vocab_size, embedding_dim, hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练数据
    train_x, train_y = build_sample(vocab, train_sample, str_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y) # 计算loss model.forward(x,y)
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        # 保存模型
        torch.save(model.state_dict(), "model.bin")
        # 保存词表
        writer = open("vocab.json", "w", encoding="utf8")
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
        writer.close()
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
    # test_strings = ["一五节快劳动乐"]
    # model_path = "model.bin"
    # vocab_path = "vocab.json"
    # predict(model_path, vocab_path, test_strings)

