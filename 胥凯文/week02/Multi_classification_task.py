import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class multi_classification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(multi_classification, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred

def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y) 

def evaluate( model ): 
    model.eval() 
    test_sample_num = 100 
    x, y = build_dataset(test_sample_num) 
    class_counts = torch.bincount(y, minlength=5)
    print("本次预测集中各类别样本数：", class_counts.tolist())
    correct, wrong = 0, 0 
    with torch.no_grad(): 
        y_pred = model(x)
        for i, (y_p, y_t, data) in enumerate(zip(y_pred, y, x)):
            pred_class = torch.argmax(y_p).item()
            if pred_class == y_t.item(): 
                correct += 1
            else: 
                wrong += 1 
                print(f"错误预测 #{i+1}:")
                print(f"输入数据: {data.tolist()}")
                print(f"真实标签: {y_t.item()}")
                print(f"预测结果: {pred_class}")
                print(f"模型输出: {y_p.tolist()}")
                print("-" * 60)

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong))) 
    return correct / (correct + wrong)

def main():
    epoch_num = 100
    batch_size = 20
    train_sample = 10000
    input_size = 5
    num_classes = 5
    learning_rate = 0.001

    model = multi_classification(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log=[]
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.legend()
    plt.savefig("training_result.png")
    return

def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = multi_classification(input_size, num_classes)    
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
        print("-" * 60)
        for vec, res in zip(input_vec, result):
            pred_class = torch.argmax(res).item()
            print(f"输入：{vec}, 预测结果: {pred_class}")
        print("-" * 60)

if __name__ == "__main__":
    main()
    test_vec = [[0.86188838,0.58855548,0.27233023,0.10965117,0.41581363],   
                [0.01126027,0.2470558,0.07383749,0.23385103,0.08679923],
                [0.59107812,0.62984888,0.42401205,0.40745962,0.7537498 ],
                [0.34939678,0.74418032,0.19661039,0.89827735,0.97360889],
                [0.36452281,0.33619154,0.01281875,0.56922397,0.20749388],
                [0.12309462,0.84128805,0.97913024,0.69720922,0.21416068],
                [0.9892495,0.97009857,0.18373845,0.2050438,0.50362554],
                [0.61145565,0.81017807,0.70994156,0.40363325,0.4437242],
                [0.11906734,0.18346966,0.19999348,0.02153939,0.30359786],
                [0.65625453,0.63779863,0.94826999,0.74412394,0.68712044]]
    predict("model.bin", test_vec)
