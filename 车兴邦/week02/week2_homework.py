import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
规律：x是一个5维向量，哪维数字最大就属于第几类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):  # 新增num_classes参数
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  
        self.activation = nn.Softmax(dim=1)  #softmax激活函数
        self.loss = nn.CrossEntropyLoss()  # 多分类标准损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        y_pred = self.activation(x)  # (batch_size, 5) -> 概率分布
        if y is not None:
            return self.loss(x, y)  
        else:
            return y_pred  # 返回概率分布


# 生成一个样本
# 随机生成一个5维向量，找出最大值所在的维度作为类别
def build_sample():
    x = np.random.random(5)  # 5维随机向量
    # 找出最大值所在的索引(0-4)
    max_index = np.argmax(x)
    return x, max_index  # 返回(特征, 类别索引)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y) 
    return torch.FloatTensor(X), torch.LongTensor(Y)  


# 测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计每个类别的样本数
    class_counts = [0] * 5
    for label in y:
        class_counts[label] += 1
    print("本次预测集各类别样本数:", class_counts)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回概率分布 (100, 5)
        # 取预测概率最大的类别作为预测结果
        y_pred_class = torch.argmax(y_pred, dim=1)  # (100,)
        
        # 与真实标签对比
        for y_p, y_t in zip(y_pred_class, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 新增: 类别数量
    learning_rate = 0.01  # 学习率
    
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")
    
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
