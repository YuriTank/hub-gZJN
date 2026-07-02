# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()

        self.linear = nn.Linear(input_size, input_size)

        # 交叉熵损失
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # ===== 前向传播 =====

        # logits（未归一化分数）
        # shape: (batch, num_classes)
        logits = self.linear(x)

        # ===== 如果有标签：训练 =====
        if y is not None:
            # CrossEntropyLoss:
            # 内部自动做：
            #   softmax + log + NLL
            loss = self.loss(logits, y)
            return loss

        # ===== 否则：预测 =====
        else:
            return logits


def build_sample(input_size):
    x = np.random.random(input_size)

    # 标签 = 最大值索引
    y = np.argmax(x)

    return x, y

# 构造数据集
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []

    for _ in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)  
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model, input_size):
    model.eval()

    test_sample_num = 200
    x, y = build_dataset(test_sample_num, input_size)

    correct = 0

    with torch.no_grad():
        logits = model(x)

        pred = torch.argmax(logits, dim=1)

        for p, t in zip(pred, y):
            if int(p) == int(t):
                correct += 1

    acc = correct / test_sample_num
    print("准确率：", acc)
    return acc

def main():
    # ===== 参数 =====
    epoch_num = 20
    batch_size = 32
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01

    model = TorchModel(input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_x, train_y = build_dataset(train_sample, input_size)

    log = []

    for epoch in range(epoch_num):
        model.train()
        losses = []

        for i in range(0, train_sample, batch_size):
            x = train_x[i:i+batch_size]
            y = train_y[i:i+batch_size]

            loss = model(x, y)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        acc = evaluate(model, input_size)
        log.append((acc, avg_loss))

    plt.plot([l[0] for l in log], label="acc")
    plt.plot([l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()