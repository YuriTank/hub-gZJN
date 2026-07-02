"""【本周第二周作业：】
尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
"""

import numpy as np
import torch
import torch.nn as nn

# 生成一个随机样本并返回其最大值索引
def build_sample():
    x = np.random.randint(0, 10, 5)  # 生成一个包含5个随机数的数组
    return x, x.argmax()

# 随机生成训练样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size, output_size): 
        super(TorchModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  

    def forward(self, x):
        x = self.fc(x)  # 线性变换, shape: (batch_size, 5)
        return x

# 定义损失函数
criterion = nn.CrossEntropyLoss()
def loss_fn(y_pred, y_true):
    return criterion(y_pred, y_true)
    
# 测试模型
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # y 现在是 Tensor，需要转成 numpy 才能用 np.unique
    print(f'本次预测集中样本分布为，{np.unique(y.numpy(), return_counts=True)}')
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # 需要取出预测概率最大的那个索引，才是预测类别
        predict_classes = torch.argmax(y_pred, dim=1) 
        
        for y_p, y_t in zip(predict_classes, y):  
            if y_p == y_t:
                correct += 1  
            else:
                wrong += 1 
                
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 训练模型
def main():
    # 配置参数
    epoch_num = 50
    batch_size = 20
    train_sample = 5000
    input_size = 5
    output_size = 5  
    learning_rate = 0.01

    # 生成训练数据
    X, Y = build_dataset(train_sample)
    
    # 初始化模型 (传入输出维度5)
    model = TorchModel(input_size, output_size)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    for epoch in range(epoch_num):
        for i in range(0, train_sample, batch_size):
            # 获取当前批次数据
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]
            
            optimizer.zero_grad() # 【规范调整】：梯度清零放在前向传播前面
            
            # 前向传播   
            y_pred = model.forward(x_batch)
            # 计算损失 (使用外部定义的函数)
            loss = loss_fn(y_pred, y_batch)   
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
        print('Epoch [{}/{}], Loss: {:.4f},'.format(epoch+1, epoch_num, loss.item()))
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    #输出权重系数
    print(model.state_dict())
    # 测试本轮模型结果
    acc = evaluate(model)  
    print('Accuracy: {:.4f}\n'.format(acc))


if __name__ == '__main__':
    main()

