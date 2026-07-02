import torch.optim
from torch.utils.data import DataLoader

from MyDataset import MyDataSet
from torch.optim import Adam
from DivNet import LSTMClsModel, RNNClsModel
import matplotlib.pyplot as plt

"""
新闻内容分类任务
使用双向LSTM（备选双向RNN)
"""
def main():
    epochs = 100
    batch_size = 64
    learning_rate = 0.005
    hidden_size = 256
    embedding_size = 256

    #数据加载
    dataset = MyDataSet(r'./data/train_tag_news.json',
                        r'./data/valid_tag_news.json',
                        r'./data/chars.txt')
    data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    model = LSTMClsModel(len(dataset.vocab), embedding_size, hidden_size, len(dataset.labels))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    logs = []
    loss_items = []
    for epoch in range(epochs):
        model.train()
        dataset.set_mode(0)
        for batch_idx, (data, target) in enumerate(data_loader):
            loss = model(data, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_items.append(loss.item())

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                model.eval()
                dataset.set_mode(1)
                for data, target in data_loader:
                    acc = torch.mean((model(data).argmax(dim=1) == target).float())
            avg_loss = sum(loss_items) / len(loss_items)
            print(f'Epoch {epoch + 1}, loss: {sum(loss_items) / len(loss_items):.8f}, acc: {acc:.2f}')
            loss_items = []
            logs.append({'epoch': epoch + 1, 'loss': avg_loss, 'acc': acc})
    show(logs)

def show(logs):
    epochs = [item['epoch'] for item in logs]
    losses = [item['loss'] for item in logs]
    accs = [item['acc'] for item in logs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-o', label='Loss', linewidth=2, markersize=4)
    plt.plot(epochs, accs, 'r-s', label='Accuracy', linewidth=2, markersize=4)
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('loss and accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == '__main__':
    main()