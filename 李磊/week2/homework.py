### 训练一个模型寻找这样一个规律：输入一个向量(5,)判断最大值的索引

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

model_path = "model\\mutile_calssif.bin"    # 运行该程序路径：week2文件夹下，否则需要改该路径

# 模型构建：多分类模型
class MultiClassifModel(nn.Module):
    def __init__(self, input_size=5, output_size=5):
        """
        模型构造
        Args:
            input_size: 输入样本 (input_size,)
            output_size: 输出样本 (output_size,)
        """
        super(MultiClassifModel, self).__init__()   # 继承父类
        self.fc = nn.Linear(input_size, output_size)    # 全连接层
        self.sm = nn.Softmax(dim=1) # Softmax层：预测用，选择第二个维度softmax,第一个维度是batch
        self.ce = nn.CrossEntropyLoss(reduction="mean") # 均值交叉熵损失函数
    
    def forward(self, x, labels=None):
        """
        前向传播方法（支持训练和预测两种模式）
        Args:
            x: 一个批次的数据，形状为 (batch_size, input_size)
            labels: 标签张量，形状为 (batch_size,)
               - 如果提供 labels,进入训练模式，返回损失值
               - 如果不提供 labels,进入预测模式,返回概率分布
        
        Returns:
            当 labels=None 时：返回预测值 (batch_size, output_size)
            当 labels 提供时：返回损失值（标量）
        """
        
        logits = self.fc(x) # 全连接层
        
        # 如果提供了标签，计算并返回损失值
        if labels is not None:
            loss = self.ce(logits, labels)  # ce内部会对logits进行softmax
            return loss
        
        # 如果没有标签，返回预测值
        probs = self.sm(logits)
        return probs


# 创建样本
def build_dataset(total_sample=1000):
    def build_sample():
        x = np.random.random(5)
        y = np.argmax(x)    # 选择最大值的标签
        return x,y
    
    X = []
    Y = []
    for i in range(total_sample):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 构建torth张量

# 评估模型准确率
def evaluate(moldel):
    moldel.eval()   # 设置模型为评估模式
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    correct,wrong = 0,0
    with torch.no_grad():   # 禁用梯度计算，节省时间
        y_pred = moldel(x)
        for y_p,y_t in zip(y_pred,y):
            if torch.argmax(y_p) == y_t: # y_p是向量，用argmax取最大值
                correct += 1
            else:
                wrong += 1
    print("准确率：%f" % (correct / (correct + wrong)))
    return correct / (correct + wrong)

# 模型训练
def model_train():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 50000  # 样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    model = MultiClassifModel(input_size)   # 构建模型对象
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate) # 选择优化器
    train_x, train_y = build_dataset(train_sample)
    log = []
    for epoch in range(epoch_num):
        model.train()   # 训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model.forward(x,y)   # 求loss
            loss.backward()             # 计算梯度
            optim.step()                # 反向传播更新参数
            optim.zero_grad()           # 重置梯度:PyTorch默认会累积梯度，如果不清零，每次迭代的梯度会叠加
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc,np.mean(watch_loss)])

    print("模型参数:",model.state_dict())
    torch.save(model.state_dict(), model_path) # 保存模型参数
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def model_predict():
    #加载模型
    input_size = 5
    model = MultiClassifModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print("模型参数:",model.state_dict())

    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,100,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.00526,1,0.995,0.41567412,0.1358894]]
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(test_vec))  # 模型预测
    for vec, res in zip(test_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (vec, torch.argmax(res), res))

if __name__ == "__main__":
    model_train()
    # model_predict()