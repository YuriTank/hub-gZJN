# coding:utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
任务描述：
x是5维向量，哪个位置数字最大，就属于第几类（1-5类）
使用：交叉熵损失（Softmax）
"""

# ---------------------- 初始模型，只有一层线性层，经测试准确率只有10%多一些---------------------

# class TorchModel(nn.Module):
#     def __init__(self, input_size, num_classes=5):   # input_size: 输入特征的维度；num_classes: 输出的类别数
#         super(TorchModel, self).__init__()
#         # 输出 5 个值，对应 5 个类别
#         self.linear = nn.Linear(input_size, num_classes)
#         # 多分类用 Softmax
#         self.activation = nn.Softmax(dim=1)   # dim=1 表示对每行进行 softmax，输出每行的概率分布
#         # 多分类损失：交叉熵
#         self.loss = nn.CrossEntropyLoss()

#     # 前向传播
#     def forward(self, x, y=None):
#         x = self.linear(x)    # [batch,5] → [batch,5]
#         y_pred = self.activation(x)  # 输出5个概率

#         if y is not None:
#             # PyTorch 交叉熵的官方正确用法———PyTorch 的 nn.CrossEntropyLoss () 内部自带 Softmax，所以我们直接传入线性层的输出（logits）即可，不需要再经过 softmax 处理。
#             logits = self.linear(x)
#             return self.loss(logits, y.long())   #y.long() 转换为 LongTensor，适用于分类标签。
#         # 什么是LongTensor？它是 PyTorch 中的一种数据类型，专门用于存储整数数据。对于分类任务，标签通常是整数形式的类别索引，因此需要将标签转换为 LongTensor 以确保与损失函数兼容。
#         else:
#             return y_pred


# ---------------------- 修改后的模型，加入隐藏层后准确率提升到了90%以上 ---------------------
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)  # 隐藏层，这里的16是一个超参数，可以根据需要调整，表示隐藏层的神经元数量。输入层有input_size个神经元，输出层有num_classes个神经元。
        # 可以任意设置吗？是的，隐藏层的神经元数量是一个超参数，可以根据具体问题的复杂度和数据量来调整。一般来说，增加隐藏层的神经元数量可以提升模型的表达能力，但也可能导致过拟合。因此，在选择隐藏层神经元数量时，需要进行实验和调优，以找到适合当前任务的最佳配置。
        self.activation = nn.ReLU()   # 隐藏层的激活函数
        self.layer2 = nn.Linear(16, num_classes)
        self.loss = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss () 内部自带 Softmax

    def forward(self, x, y=None):  # 前向传播，输入 x 是一个二维张量，包含多个输入样本，每行是一个5维向量；y 是对应的标签，如果提供了标签，则计算损失并返回；如果没有提供标签，则返回预测的概率分布。
        x = self.activation(self.layer1(x))
        logits = self.layer2(x)
        if y is not None:
            return self.loss(logits, y.long())
        else:
            return torch.softmax(logits, dim=1)

# --------------------- 生成样本：哪个数最大，就是第几类 ---------------------
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 0,1,2,3,4 对应第1-5个数，一维数组不用指定维度dim
    # np.max (x) → 返回最大值是几
    # np.argmax (x) → 返回最大值在第几个位置

    return x, max_index

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# --------------------- 评估准确率 ---------------------
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    wrong = 0

    with torch.no_grad():
        y_pred = model(x)  # [100,5] 每行是5个类别的概率
        pred_classes = torch.argmax(y_pred, dim=1)  # 取概率最大的类别标签，0-4 对应第1-5类；dim=1 表示对每行进行操作，返回每行最大值的索引，总共有100行，输出100个索引值。

        for p, t in zip(pred_classes, y):
            if p == t:
                correct +=1
            else:
                wrong +=1

    print("正确预测：%d, 正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

# --------------------- 训练主函数 ---------------------
def main():
    epoch_num = 20
    batch_size = 10
    train_sample = 15000
    input_size = 5
    learning_rate = 0.01

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        total_loss = []

        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index*batch_size : (batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size : (batch_index+1)*batch_size]

            loss = model(x, y)  # 计算当前批次的损失值，模型的 forward 方法会根据是否提供标签 y 来决定是返回预测结果还是计算损失值。当提供了标签 y 时，模型会计算交叉熵损失并返回该损失值。
            loss.backward()  # 反向传播，计算损失值相对于模型参数的梯度
            optim.step()  # 更新模型参数，使用优化器根据计算得到的梯度来调整模型的权重，以最小化损失函数
            optim.zero_grad()  # 清除之前计算的梯度，避免梯度累积导致更新错误
            total_loss.append(loss.item())  # loss.item() 将损失值从张量转换为 Python 的浮点数，方便后续的平均计算和日志记录

        print("===== 第%d轮, 平均loss:%.4f =====" % (epoch+1, np.mean(total_loss)))
        acc = evaluate(model)
        log.append([acc, np.mean(total_loss)])

    torch.save(model.state_dict(), "model_5class.bin")

    model_weights = torch.load("model_5class.bin")
    print(model_weights)   # 结果如下


    plt.plot([l[0] for l in log], label="acc")
    plt.plot([l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# --------------------- 预测：输出 1~5 类别 ---------------------
def predict(model_path, input_vec):
    model = TorchModel(5)   # 输入是5维向量，输出是5个类别的概率
    model.load_state_dict(torch.load(model_path))   # 加载之前训练好的模型参数
    model.eval()  # 将模型设置为评估模式，这样在预测时会关闭 dropout 和 batch normalization 等训练时特有的行为，确保模型在推理阶段的表现与训练阶段一致。

    with torch.no_grad():
        pred = model(torch.FloatTensor(input_vec))  # 模型预测，输入是一个二维张量，包含多个5维向量，每行对应一个输入样本。模型会输出一个二维张量，每行包含5个类别的概率值，表示每个输入样本属于各个类别的概率分布。

    for vec, prob in zip(input_vec, pred):
        max_class = torch.argmax(prob).item()  # 0-4，torch.argmax(prob).item()返回概率值最大的类别索引，.item()将其转换为Python的整数类型，不加.item()会返回一个张量对象，如tensor(2)，加了.item()后会返回一个普通的整数值，如2。
        final_class = max_class + 1            # 1-5
        print("输入：%s → 最大位置：第%d类 | 概率：%s" % (vec, final_class, np.round(prob.numpy(),3))) 
        # np.round(prob.numpy(),3)) 将概率值四舍五入到小数点后三位，prob.numpy()将PyTorch张量转换为NumPy数组，以便使用NumPy的round函数进行四舍五入操作。通过这种方式，我们可以更清晰地看到每个类别的概率值，并且知道哪个类别具有最高的概率，从而做出最终的分类决策。

if __name__ == "__main__":
    main()
    test_vec = [
        [0.8, 0.2, 0.3, 0.1, 0.4],   # 第1个最大 → 类别1
        [0.1, 0.9, 0.3, 0.2, 0.4],   # 第2个最大 → 类别2
        [0.2, 0.1, 0.8, 0.3, 0.2],   # 第3个最大 → 类别3
        [0.1, 0.2, 0.3, 0.9, 0.1],   # 第4个最大 → 类别4
        [0.1, 0.2, 0.1, 0.2, 0.9]    # 第5个最大 → 类别5
    ]
    predict("model_5class.bin", test_vec)



# ---------------------- 备注 ---------------------
'''
nn.Linear(5, 16)
输入维度 = 5（你输入的 5 个数字）
输出维度 = 16（隐藏层 16 个神经元）
PyTorch 内部的权重形状是：(out_features, in_features)
也就是：(16, 5)
所以，第一层的权重形状的确是 (16, 5)，而不是 (5, 16)。这是因为在 PyTorch 中，线性层的权重矩阵是按照输出特征数（out_features）和输入特征数（in_features）的顺序排列的。
隐藏层输出 = 输入 × W.T + b
输入形状：(batch, 5)
权重形状：(16, 5)
权重转置后：(5, 16)
输出形状：(batch, 16)，正好得到 16 维隐藏层输出！

输入 (5)
   ↓
Linear(5,16) → 权重(16,5) → 输出(16)
   ↓
ReLU
   ↓
Linear(16,5) → 权重(5,16) → 输出(5)
   ↓
Softmax → 5个概率

所以，运行后可以看到第1层的权重矩阵是（16，5），第2层的权重矩阵是（5，16）。这是 PyTorch 线性层的标准权重形状，确保了输入和输出之间的正确矩阵乘法运算。
'''
