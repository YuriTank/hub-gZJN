import torch
import torch.nn as nn

dim = 5
batch_size = 32
epochs = 100
lr = 0.01

# 生成数据
def generate_data(n):
    X = torch.randn(n, dim)
    y = torch.argmax(X, dim=1)
    return X, y

# 模型：线性层
model = nn.Linear(dim, dim)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(epochs):
    X, y = generate_data(batch_size)
    pred = model(X)
    loss = criterion(pred, y)
    
    # 梯度清零
    if model.weight.grad is not None:
        model.weight.grad.zero_()
        model.bias.grad.zero_()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    with torch.no_grad():
        model.weight.data -= lr * model.weight.grad
        model.bias.data -= lr * model.bias.grad
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 测试
with torch.no_grad():
    test_X, test_y = generate_data(100)
    pred = model(test_X)
    acc = (torch.argmax(pred, dim=1) == test_y).float().mean()
    print(f'\n测试准确率: {acc.item():.4f}')
