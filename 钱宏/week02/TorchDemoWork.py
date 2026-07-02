# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib

matplotlib.use('TkAgg')

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数（多分类任务）  内部已包含Softmax

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，进行分类。
# 如果向量第一个最大，分类一
# 第二个最大，分类二
# 第三个最大，分类三
# 模型输出5个类别（索引0-4）
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 返回最大值索引
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 直接 append 标量，不要包成列表 [y]
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签用LongTensor 存储整数类别标签


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct, wrong = 0, 0
    error_indices = []  #  记录错误样本的索引
    with torch.no_grad(): # 禁用梯度计算，用于推理/评估阶段。
        y_pred = model(x)  # 模型预测 model.forward(x)
        predictions = torch.argmax(y_pred, dim=1)  #  取最大值索引，维度

        for idx, (y_p, y_t) in enumerate(zip(predictions, y)):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
                error_indices.append(idx)  # 记录错误索引
    print("预测值：%s，真实值：%s" % (predictions, y))
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))

    if error_indices:
        print("\n预测错误的样本索引及详情：")
        for idx in error_indices:
            print(f"  索引 {idx}: 输入={x[idx].numpy()}, 预测={predictions[idx].item()}, 真实={y[idx].item()}")

    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() # 设置模型为训练模式（启用dropout、batchnorm等训练特定行为）
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
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
    with torch.no_grad():
        logits = model.forward(torch.FloatTensor(input_vec))
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)

    for vec, pred, prob in zip(input_vec, predictions, probabilities):
        max_prob = torch.max(prob).item()
        print("输入：%s, 预测类别：%d, 置信度：%f" % (vec, int(pred), max_prob))


if __name__ == "__main__":
     main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
