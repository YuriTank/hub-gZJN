import numpy as np
import torch.nn as nn
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 生成1000个四维数据
X = np.random.rand(1000, 4)

# 生成他们标签
y = np.argmax(X, axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 构建模型
class TorchModel(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, matrix:torch.tensor):
        x = self.layer1(matrix)
        y = self.layer2(x)
        return y

def plot_confusion_matrix(model:nn.Module, X_test:torch.tensor, y_test:torch.tensor, save_path:str='confusion_matrix.png'):
    """ 绘制混淆矩阵 """
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test)
        y_pred = torch.argmax(y_pred_logits, dim=1)

    y_true_np = y_test.numpy()
    y_pred_np = y_pred.numpy()

    cm = confusion_matrix(y_true=y_true_np, y_pred=y_pred_np)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xlabel('Predict')
    plt.ylabel('true')
    plt.title('Confusion_matrix')
    plt.xticks([0, 1, 2, 3])
    plt.yticks([0, 1, 2, 3])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {save_path}")
    plt.close()

def plot_training_history(history:dict, save_path:str='training_history.png'):
    """ 绘制训练过程中loss和accuracy的变化 """
    epochs_range = range(len(history['train_loss']))

    plt.figure(figsize=(12, 5))

    # 绘制loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # 绘制accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['test_accuracy'], label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存到: {save_path}")
    plt.close()
def train(X_train:torch.tensor, y_train:torch.tensor, input_size:int, hidden_size:int, output_size:int, epochs: int,
          X_test:torch.tensor, y_test:torch.tensor, batch_size:int=32):
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TorchModel(input_size, hidden_size, output_size)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    # 记录训练历史
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        # 批量训练
        for batch_X, batch_y in train_loader:
            # 清空梯度
            optimizer.zero_grad()
            # 预测数据
            y_pred = model(batch_X)
            # 计算训练损失
            train_loss = loss_fn(y_pred, batch_y)
            # 反向传播
            train_loss.backward()
            # 更新参数
            optimizer.step()
            epoch_train_loss += train_loss.item()

        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(train_loader)

        # 在测试集上评估
        with torch.inference_mode():
            model.eval()
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
            # 计算准确率
            test_pred_labels = torch.argmax(test_pred, dim=1)
            test_accuracy = (test_pred_labels == y_test).sum().item() / len(y_test)
            model.train()

        # 记录历史数据
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(test_loss.item())
        history['test_accuracy'].append(test_accuracy)

        if epoch%10 == 0:
            print(f"Epoch:{epoch} | train_loss:{avg_train_loss:.5f} | test_loss:{test_loss:.5f} | test_acc:{test_accuracy:.4f}")

    return model, history

def load_and_predict():
    """ 加载并预测 """
    model = TorchModel(input_size=4, hidden_size=10, output_size=4)

    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

def main(argc: int, argv: list[str]) -> int:
    print(f"参数个数: {argc}")
    print(f"参数列表: {argv}")

    if argc > 1 and argv[1] == "train":
        print("开始训练...")
        model, history = train(X_train=X_train, y_train=y_train, input_size=4, hidden_size=10,
                     output_size=4, epochs=100, X_test=X_test, y_test=y_test)
        torch.save(model.state_dict(), 'model.pth')
        print("模型已保存到 model.pth")
        plot_training_history(history)
        return 0

    elif argc > 1 and argv[1] == "predict":
        print("加载模型进行预测...")
        test_sample = torch.tensor([[0.1, 0.9, 0.3, 0.2]], dtype=torch.float32)
        model_load = load_and_predict()
        with torch.no_grad():
            prediction = model_load(test_sample)
            predicted_class = torch.argmax(prediction, dim=1)
            print(f"预测类别: {predicted_class.item()}")
        return 0

    else:
        model, history = train(X_train=X_train, y_train=y_train, input_size=4, hidden_size=10,
                     output_size=4, epochs=100, X_test=X_test, y_test=y_test)
        torch.save(model.state_dict(), 'model.pth')
        print("模型已保存")

        test_sample = torch.tensor([[0.1, 0.9, 0.3, 0.2]], dtype=torch.float32)
        with torch.no_grad():
            prediction = model(test_sample)
            predicted_class = torch.argmax(prediction, dim=1)
            print(f"预测类别: {predicted_class.item()}")

        plot_training_history(history)
        plot_confusion_matrix(model=model, X_test=X_test, y_test=y_test)
        return 0

if __name__ == "__main__":
    argc = len(sys.argv)
    argv = sys.argv
    exit_code = main(argc, argv)
    sys.exit(exit_code)
    
