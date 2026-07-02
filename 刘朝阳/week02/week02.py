import torch
import torch.nn as nn
import numpy as np

# 数据生成
def build_sample(input_dim):
    x = np.random.random(input_dim) # 随机生成一个向量
    y = np.argmax(x)  # 最大值的索引作为类别标签
    return x, y

def build_dataset(total_samples, input_dim):
    X = []
    Y = []
    for _ in range(total_samples):
        x, y = build_sample(input_dim)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 定义模型
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)  # 输出层不加激活，CrossEntropyLoss内部有softmax
        return x

# 训练
def train(input_dim=5, epochs=20):
    num_classes = input_dim
    hidden_dim = input_dim * 2
    batch_size = 32
    learning_rate = 0.01
    
    print(f"向量维度: {input_dim}, 类别数: {num_classes}")
    
    # 准备数据
    train_x, train_y = build_dataset(5000, input_dim)
    test_x, test_y = build_dataset(500, input_dim)
    
    # 初始化模型、损失函数、优化器
    model = Classifier(input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()  # 多分类用交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]
            
            # 前向传播
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 测试准确率
        model.eval()
        with torch.no_grad():
            test_output = model(test_x)
            pred = torch.argmax(test_output, dim=1)
            acc = (pred == test_y).float().mean()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.2%}")
    
    return model, input_dim

# 预测
def predict(model, input_dim, num_samples=5):
    model.eval()
    test_samples = []
    for _ in range(num_samples):
        x = np.random.random(input_dim)
        test_samples.append(x)
    
    with torch.no_grad():
        x = torch.FloatTensor(test_samples)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        prob = torch.softmax(output, dim=1)
        
        for vec, p, prob_vec in zip(test_samples, pred, prob):
            true_class = np.argmax(vec)
            print(f"输入向量: {vec}")
            print(f"真实类别: {true_class}, 预测类别: {p.item()}, 概率分布: {prob_vec.numpy()}")
            print("---")

# 运行
INPUT_DIM = 5 # 向量维度，同时也是类别数
    
model, input_dim = train(input_dim=INPUT_DIM)
predict(model, input_dim)
