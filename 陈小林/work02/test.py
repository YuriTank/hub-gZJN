import torch

from DivNet import MaxNet


if __name__=="__main__":
    #构造测试数据
    data_size = 1000
    data = torch.randn(data_size, 5)
    labels = torch.max(data, dim=1, keepdim=False).indices

    #定义训练参数
    epochs = 1000
    batch_size = 32
    #创建网络
    net = MaxNet(5,5)
    optimize = torch.optim.Adam(net.parameters(), lr=0.01)
    log = []
    accs = []
    net.train()
    for epoch in range(epochs):
        for b in range(0, data_size, batch_size):
            last = b + batch_size
            if last >= data_size:
                last = -1
            x = data[b:last,:]
            y = labels[b:last]
            loss = net(x, y)
            loss.backward()
            log.append(loss.item())
            optimize.step()
            optimize.zero_grad()
            with torch.no_grad():
                net.eval()
                y_pred = net(x)
                acc = torch.mean((torch.max(y_pred, dim=1, keepdim=False).indices == y).float())
                accs.append(acc.item())

        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss: {torch.mean(torch.tensor(log))}, acc: {torch.mean(torch.tensor(accs))}')

    #效果展示
    net.eval()
    y_pred = net(torch.tensor([[1,2,6,4,5]]).float())
    print(y_pred)

