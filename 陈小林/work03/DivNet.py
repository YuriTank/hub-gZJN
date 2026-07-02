import torch
from torch import nn
"""
网络定义。用于分类文本
可选模型有：双向LSTM，双向RNN
"""
class LSTMClsModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(LSTMClsModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        x = torch.cat([x[0], x[1]], dim=-1)
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        return x

class RNNClsModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        super(RNNClsModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.RNN(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        x = torch.cat([x[0], x[1]], dim=-1)
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        return x
