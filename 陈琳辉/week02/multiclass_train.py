"""
多分类任务训练：随机向量最大维度分类

任务描述：
- 输入：一个随机向量（如5维向量 [0.1, 0.8, 0.3, 0.2, 0.4]）
- 标签：向量中最大值的索引（如上述向量，最大值是0.8在索引1位置，则标签为1）
- 类别数：等于向量维度数

网络结构：
- 使用全连接神经网络 (MLP)
- 输入层：向量维度
- 隐藏层：若干层
- 输出层：类别数（由 CrossEntropyLoss 内部计算 softmax）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 保存路径：脚本所在目录
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============ 配置参数 ============
class Config:
    vector_dim: int = 5
    num_classes: int = vector_dim
    hidden_dim: int = 64
    num_epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    random_seed: int = 42


# ============ 数据集类 ============
class RandomVectorDataset(Dataset):
    """随机向量数据集"""

    def __init__(self, num_samples: int, vector_dim: int, seed: int = None):
        self.num_samples = num_samples
        self.vector_dim = vector_dim

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data = np.random.rand(num_samples, vector_dim).astype(np.float32)
        self.labels = np.argmax(self.data, axis=1).astype(np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])


# ============ 模型定义 ============
class MultiClassClassifier(nn.Module):
    """多分类全连接神经网络"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# ============ 训练函数（记录准确率） ============
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device
) -> Tuple[list, list, list, list]:
    """
    训练模型

    Returns:
        (train_loss_history, train_acc_history, test_loss_history, test_acc_history)
    """
    model.train()
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_data)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # 评估阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                logits = model(data)
                loss = criterion(logits, labels)

                test_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * correct / total

        # 记录历史
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history


# ============ 可视化函数 ============
def plot_training_curves(
    train_loss: List[float],
    train_acc: List[float],
    test_loss: List[float],
    test_acc: List[float],
):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_loss) + 1)

    # 损失曲线
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, test_loss, 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, test_acc, 'r-', label='Test Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_prediction_samples(model: nn.Module, num_samples: int, vector_dim: int, device: torch.device):
    """可视化预测样本和概率分布"""
    model.eval()

    # 生成测试样本
    np.random.seed(123)
    test_data = np.random.rand(num_samples, vector_dim).astype(np.float32)
    true_labels = np.argmax(test_data, axis=1)

    with torch.no_grad():
        inputs = torch.from_numpy(test_data).to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        _, predicted = torch.max(probs, 1)
        predicted = predicted.cpu().numpy()
        probs = probs.cpu().numpy()

    # 创建可视化
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        colors = ['#3498db'] * vector_dim
        colors[true_labels[i]] = '#2ecc71'
        if predicted[i] != true_labels[i]:
            colors[predicted[i]] = '#e74c3c'

        bars = ax.bar(range(vector_dim), probs[i], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f"True: {true_labels[i]} | Pred: {predicted[i]}",
                    fontsize=10, fontweight='bold',
                    color='#e74c3c' if predicted[i] != true_labels[i] else '#27ae60')
        ax.set_xlabel('Class', fontsize=9)
        ax.set_ylabel('Probability', fontsize=9)
        ax.set_xticks(range(vector_dim))
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # 在最大概率柱上显示数值
        max_idx = np.argmax(probs[i])
        ax.text(max_idx, probs[i][max_idx] + 0.05, f'{probs[i][max_idx]:.2f}',
               ha='center', fontsize=8, fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='True Class'),
        Patch(facecolor='#3498db', edgecolor='black', label='Other Classes'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Wrong Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Prediction Probability Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(model: nn.Module, test_loader: DataLoader, num_classes: int, device: torch.device):
    """绘制混淆矩阵"""
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_labels, all_predictions):
        cm[true_label][pred_label] += 1

    # 绘制
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    # 添加数值标签
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============ 推理函数 ============
def predict(model: nn.Module, data: torch.Tensor, device: torch.device):
    """推理函数"""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        logits = model(data)
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1)
    return predicted.cpu(), probs.cpu()


# ============ 主函数 ============
def main():
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据集
    train_dataset = RandomVectorDataset(num_samples=1000, vector_dim=Config.vector_dim, seed=42)
    test_dataset = RandomVectorDataset(num_samples=200, vector_dim=Config.vector_dim, seed=999)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # 创建模型
    model = MultiClassClassifier(
        input_dim=Config.vector_dim,
        hidden_dim=Config.hidden_dim,
        num_classes=Config.num_classes
    ).to(device)

    print(f"\n模型结构:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数总数: {total_params}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    # 训练
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50)

    results = train_model(
        model, train_loader, test_loader,
        criterion, optimizer,
        Config.num_epochs, device
    )
    train_loss, train_acc, test_loss, test_acc = results

    # 最终评估
    print("\n" + "=" * 50)
    print("最终测试集评估")
    print("=" * 50)
    print(f"最终准确率: {test_acc[-1]:.2f}%")
    print(f"最终损失: {test_loss[-1]:.4f}")

    # ============ 可视化 ============
    print("\n" + "=" * 50)
    print("生成可视化图表")
    print("=" * 50)

    plot_training_curves(train_loss, train_acc, test_loss, test_acc)
    plot_prediction_samples(model, num_samples=10, vector_dim=Config.vector_dim, device=device)
    plot_confusion_matrix(model, test_loader, Config.num_classes, device=device)

    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vector_dim': Config.vector_dim,
        'hidden_dim': Config.hidden_dim,
        'num_classes': Config.num_classes,
        'train_loss_history': train_loss,
        'test_acc_history': test_acc
    }
    torch.save(checkpoint, os.path.join(SAVE_DIR, 'multiclass_model.pth'))
    print(f"\n模型已保存到: {os.path.join(SAVE_DIR, 'multiclass_model.pth')}")

    return model, results


if __name__ == '__main__':
    main()
