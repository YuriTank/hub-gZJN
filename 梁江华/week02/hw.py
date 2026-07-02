# coding:utf8

# 解决 OpenMP 库冲突问题
import os

from torch.nn import functional

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from collections import Counter

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，尝试完成一个多分类任务的训练:哪一维数字最大就属于第几类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，输出维度为类别数

    # 当输入真实标签，返回loss值；无真实标签，返回原始分数(logits)
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return nn.functional.cross_entropy(x, y)
        else:
            return x  # 输出原始分数 (logits)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，找到最大数的地址
def build_sample():
    x = np.random.random(5)
    largest_index = 0
    largest_num = x[0]

    for i in range(1, 5):
        if x[i] > largest_num:
            largest_num = x[i]
            largest_index = i

    return x, largest_index + 1


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y - 1)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)  # int64 for cross_entropy
    return torch.from_numpy(X), torch.from_numpy(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, loss_fn):
    model.eval()
    test_sample_num = 10000
    x, y = build_dataset(test_sample_num)

    # 1. Count label occurrences
    y_list = [int(v) for v in y]
    label_count = Counter(y_list)

    print("Label Counts in Y:")
    for label in sorted(label_count.keys()):
        print(f"  Label {label + 1}: {label_count[label]} times (Class {label + 1})")  # 显示用户习惯的 1-based 标签

    correct, wrong = 0, 0
    with torch.no_grad():
        predictions = model(x)  # 模型输出的是 logits
        # 获取最高分的索引，这就是预测的类别 (0-4)
        predicted_classes = torch.argmax(predictions, dim=1)

        # 真实的类别标签已经是 0-indexed (y)
        true_classes = y.squeeze()  # 去掉多余的维度

        # 比较预测类别和真实类别
        correct_predictions = (predicted_classes == true_classes).sum().item()
        total_samples = x.size(0)

        print("正确预测个数：%d, 正确率：%.4f" % (correct_predictions, correct_predictions / total_samples))
        return correct_predictions / total_samples


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数：增加到50
    batch_size = 200  # 每次训练样本个数：增大批量大小以提高稳定性
    train_sample = 20000  # 每轮训练总共训练的样本总数：增加样本量
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率：降低学习率

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            # 计算loss，并进行反向传播
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%.6f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, nn.functional.cross_entropy)  # 测试本轮模型结果，并传入交叉熵函数
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "../week01/model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
    torch.save(model.state_dict(), "../week01/model.bin")
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
    # print(model.state_dict()) # 移除状态字典打印，避免干扰输出

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测，现在返回的是5维的logits

    # 将logits转换为预测的类别（索引）
    predicted_classes = torch.argmax(result, dim=1)

    for i, vec in enumerate(input_vec):
        predicted_index = predicted_classes[i].item()
        # 将 0-based index 转换回 1-based 类别标签
        predicted_label = predicted_index + 1

        # 计算概率分布（可选：显示概率值）
        probabilities = torch.softmax(result[i], dim=0)

        print(f"输入：{vec}, 预测类别：{predicted_label}, 概率分布：{probabilities.tolist()}")
        # 如果您只需要一个代表性的概率值，可以取预测类别的概率：
        # print(f"输入：{vec}, 预测类别：{predicted_label}, 概率值：{probabilities[predicted_index].item():.4f}")


if __name__ == "__main__":
    main()
    test_vec = [[0.88889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.99349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
                [0.349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894],
                [0.349776, 0.59416669, 0.2579291, 0.81567412, 0.1358894],
                [0.349776, 0.59416669, 0.2579291, 0.1567412, 0.8894],
                ]
    predict("../week01/model.bin", test_vec)
