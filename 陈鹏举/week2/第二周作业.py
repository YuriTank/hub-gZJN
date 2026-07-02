import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 输入5维，输出5维（对应5个类别）
        self.loss = nn.CrossEntropyLoss()                # 交叉熵损失函数（内部包含softmax）

    # 前向传播: 输入x，返回logits；若提供y则返回损失值
    def forward(self, x, y=None):
        logits = self.linear(x)          # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            # CrossEntropyLoss要求y为LongTensor，形状为(batch_size,)
            return self.loss(logits, y.squeeze().long())
        else:
            return logits                # 返回logits，预测时再取argmax

# 生成一个样本: 返回5维随机向量及其最大值的索引（0~4）
def build_sample():
    x = np.random.random(5)
    # 找出第一个最大值的索引（若并列取第一个）
    y = np.argmax(x)
    return x, y

# 生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])          # 保持列维方便后续转为tensor
    return torch.FloatTensor(X), torch.LongTensor(Y)   # 标签使用LongTensor

# 测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)   # y: (100,1)
    print("本次预测集中各类样本数量：")
    # 统计每个类别的真实个数
    y_flat = y.squeeze().numpy()
    for i in range(5):
        print("类别%d: %d" % (i, np.sum(y_flat == i)))
    correct = 0
    with torch.no_grad():
        logits = model(x)                   # (100,5)
        pred = torch.argmax(logits, dim=1)  # 取最大值的索引作为预测类别 (100,)
        y_true = y.squeeze()                # (100,)
        correct = (pred == y_true).sum().item()
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

def main():
    # 配置参数
    epoch_num = 25         # 训练轮数
    batch_size = 400         # 每次训练样本个数
    train_sample = 5000     # 每轮训练总共训练的样本总数
    input_size = 5          # 输入向量维度（同时也等于类别数）
    learning_rate = 0.1    # 学习率

    # 建立模型
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 将数据按batch分割
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        probs = torch.softmax(logits, dim=1)   # 转换为概率
        pred = torch.argmax(logits, dim=1)
    for vec, p, prob in zip(input_vec, pred, probs):
        print("输入：%s, 预测类别：%d, 各类概率：%s" % (vec, p.item(), prob.numpy()))

if __name__ == "__main__":
    main()
