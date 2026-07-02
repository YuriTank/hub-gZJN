
import numpy as np
import torch



class   myModel(torch.nn.Module):
    #初始化方法，需要输入一个矩阵
    def __init__(self, input_size):
        #输入子类和子类对象，调用父类的初始化方法__init__()
        super(myModel, self).__init__()
        #定义线性层，输入是input矩阵，输出是5维矩阵,代表五分类任务
        self.linear=torch.nn.Linear(input_size,5)
        #自定义损失函数，损失函数指定为交叉熵
        self.loss=torch.nn.functional.cross_entropy

    #前向传播函数
    #输入真实结果，直接调用损失函数；无真实结果，预测每个数字是正确答案的概率
    def forward(self,x,y=None):
        #输入矩阵x，生成对矩阵x各值的权重，或者是打分
        y_pred=self.linear(x)
        #如果有真实值y，进入训练模式，计算损失值
        if y is not None:
            return self.loss(y_pred,y)
        #没有真实值，进入预测模式，返回预测值
        else:
            return torch.softmax(y_pred,dim=1)

#训练逻辑
def build_sample():
    #生成一个一维数组,长度为5，也叫五维向量
    x=np.random.random(5)
    #获取最大值的索引
    y_index=np.argmax(x)
    return x,y_index

#生成训练数据
def build_dateset(size):
    #存储随机生成的五维向量
    X=[]
    #存储五维向量中最大值的索引值，Y中值的索引与五维向量在X中的索引一致
    Y=[]
    for i in range(size):
        #生成size次随机值
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    #按照模型要求，把X转换为浮点型向量，把Y转换为长整数型向量
    return torch.FloatTensor(X),torch.LongTensor(Y)

#评估模型准确率
def evaluate(model):
    #切换为评估模式，关闭训练时的特有的行为
    model.eval()
    #测试100次
    test_size=100
    #根据训练逻辑生成正确结果数据
    x,y=build_dateset(test_size)
    #正确结果，错误结果
    true,wrong=0,0
    #临时关闭自动求导，评估阶段不需要改变权重
    with torch.no_grad():
        # 调用模型，生成预测值
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            #y_p是一个五维向量，每个值为模型计算的概率
            if torch.argmax(y_p) == int(y_t):
                true+=1
            else:
                wrong+=1
    print(f'在{test_size}个样本中，正确预测结果有{true}个，准确率为{(true/(true+wrong)):.4%}')
    return true/(true+wrong)


def main():
    #训练轮次
    epoch_num=20
    #每次训练的样本个数
    batch_size=100
    #每轮训练总共使用的样本总数
    train_sample=5000
    #输入的向量维度，5维
    input_size=5
    #学习率
    learing_rate=0.005
    #创建模型
    model=myModel(input_size)
    #选择优化器
    optim=torch.optim.Adam(model.parameters(),lr=learing_rate)

    log=[]
    #生成训练结果集合，训练数据x和正确结果y
    trainx,trainy=build_dateset(train_sample)

    #训练过程，20轮
    for i in range(epoch_num):
        #训练模式
        model.train()
        #损失值集合
        watch_loss=[]
        #执行train_sample//batch_size，整除的轮数
        for index  in range(train_sample//batch_size):
            #一轮取batch大小的数据
            x=trainx[index*batch_size:(index+1)*batch_size]
            y=trainy[index*batch_size:(index+1)*batch_size]
            #x，y值都有，进入训练模式
            loss=model(x,y)
            #计算梯度
            loss.backward()
            #优化器更新权重
            optim.step()
            #优化器梯度归零
            optim.zero_grad()
            #收集损失值
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    #保存模型
    torch.save(model.state_dict(),'model.pt')
    #画图
    print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return

#测试函数，测试模型效果
def test(model_path,input_vec):
    #五维向量
    input_size=5
    model=myModel(input_size)
    #加载模型参数，训练好的权重
    model.load_state_dict(torch.load(model_path))
    print(f'输出模型权重：\n{model.state_dict()}')
    #测试模式
    model.eval()
    #不更新梯度
    with torch.no_grad():
        #调用模型得到预测值
        result=model.forward(torch.FloatTensor(input_vec))
    for vec,res in zip(input_vec,result):
        print(f'输入值：{vec}，预测类别：{torch.argmax(res)},预测概率：{res}')

if __name__=='__main__':
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    test('model.pt',test_vec)
