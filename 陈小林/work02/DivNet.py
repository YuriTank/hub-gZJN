import torch.nn as nn

class MaxNet(nn.Module):
    """
    找出最大值 2层线型层，损失交叉熵
    """
    def __init__(self, input_size:int, num_classes:int):
        super(MaxNet, self).__init__()
        self.lc1 = nn.Linear(in_features=input_size, out_features=256)
        self.lc2 = nn.Linear(in_features=256, out_features=num_classes)
        self.cross_entropy = nn.CrossEntropyLoss()
        #结果归一化
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y=None):
        x = self.lc1(x)
        x = self.lc2(x)

        if y is not None:
            return self.cross_entropy(x, y)
        return self.softmax(x)