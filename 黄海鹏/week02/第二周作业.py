import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)
        # self.activation = torch.softmax
        self.loss = nn.CrossEntropyLoss() # 交叉熵损失函数 自带softmax

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x
        
# 随机生成一个5维向量
def build_sample():
    x = np.random.random(5)

    max_value = np.argmax(x)
    return x, max_value

# 随机生成一批样本
# 正负样本均匀生成 
def build_dataset(total_sample_num):
    # X 为输入向量
    X = []
    # Y 为标签
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    # 测试模型
    model.eval()
    # 测试样本数量
    test_sample_num = 100

    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    # 
    correct, wrong = 0, 0
    #
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # 预测值y_p是一个长度为5的向量，取最大值的索引作为预测类别
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测：%d，错误预测：%d" % (correct, wrong))
    return correct / (correct + wrong) if correct + wrong > 0 else 0


def main():
    # 配置参数
    epoch_num = 100 # 训练轮数
    batch_size = 20 # 每轮训练样本个数
    train_sample = 5000 # 每轮训练总共训练的样本总数
    input_size = 5 # 输入向量维度
    learning_rate = 0.01 # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 获取一个batch的训练数据
            # train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            # 1. 计算损失
            loss = model(x, y) 
            # 2. 计算梯度
            loss.backward()
            # 3. 权重更新
            optim.step()
            # 4. 梯度归零
            optim.zero_grad()

            # 画图
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        # 1. 获取原始输出
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # 2. 计算预测类别  
        predicted = torch.argmax(result, dim=1)
        probabilities = torch.softmax(result, dim=1)
    for i, vec in enumerate(input_vec):
        pred = predicted[i].item()  # 获取预测类别索引
        prob = probabilities[i][pred].item()  # 获取预测类别的概率值
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred, prob))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.98889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.97482528,0.13625847,0.34675372,0.19871392],
                [0.19349776,0.59416669,0.92579291,0.41567412,0.9358894]]
    predict("model.bin", test_vec)
