import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, out_class):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, out_class)  # 线性层，分类任务
        self.activation = nn.Softmax(dim=1)  # nn.Softmax() Softmax多分类输出层，输出概率分布
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        y_pred = self.activation(x)  # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
        

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, size):
    model.eval()
    test_sample_num = 100
    x = torch.randn(test_sample_num, size)# 随机生成一批样本
    y = torch.argmax(x, dim=1)#dim=1: 沿着行方向（跨类别）找最大值
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            y_p_maxindx = torch.argmax(y_p)
            if int(y_p_maxindx) == int(y_t):
                correct += 1  # 分类判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)  

def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.005  # 学习率
    out_size = 5#输出几分类
    # 建立模型
    model = TorchModel(input_size, out_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    # train_x, train_y = build_dataset(train_sample)
    train_x = torch.randn(train_sample, out_size)
    train_y = torch.argmax(train_x, dim=1)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入  
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())#
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, input_size)  # 测试本轮模型结果
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
    out_size = 5
    model = TorchModel(input_size, out_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())#获取模型或优化器的所有可学习参数（权重和偏置）及其对应的数值的方法

    model.eval()  # 测试模式,用于将模型设置为评估模式的方法
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        max_prob = torch.max(res).item()
        max_prob_index = torch.argmax(res).item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, max_prob_index, max_prob))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec = [[1,3,2,3.2,4],
                [2,1,3,0,1.5],
                [5,3,4,2,1],
                [7,9,6,3,2]]
    predict("model.bin", test_vec)
