from torch.utils.data import Dataset
import pandas as pd
import torch
"""
语料使用title训练 content太长
"""
class MyDataSet(Dataset):
    def __init__(self, train_path:str, valid_path:str, vocab_path):
        super(MyDataSet, self).__init__()
        self.mode = 0
        self.labels = {}
        self.load_vocab(vocab_path)
        self.train_data, self.train_data_max_len = self.load_data(train_path)
        self.valid_data, self.valid_data_max_len = self.load_data(valid_path)

    def load_vocab(self, vocab_path):
        with open(vocab_path, encoding='utf-8', mode='r') as f:
            self.vocab = {}
            with open(vocab_path, encoding='utf-8', mode='r') as f:
                for index, line in enumerate(f):
                    self.vocab[line.strip()] = index

    def load_data(self, path:str):
        """
        加载词表，加载数据集
        """
        data = pd.read_json(path, lines=True)
        res = []
        max = 0
        for index, row in data.iterrows():
            tag, title, context = row
            #标签初始化
            if tag not in self.labels:
                self.labels[tag] = len(self.labels)
            x = [self.vocab.get(char, self.vocab['[UNK]']) for char in title]
            item = {}
            item['x'] = x
            item['y'] = self.labels[tag]
            item['title'] = title
            if len(x) > max:
                max = len(x)
            res.append(item)
        for item in res:
            while len(item['x']) < max:
                item['x'].append(self.vocab['[UNK]'])
            item['x'] = torch.LongTensor(item['x'])
        return res, max

    def set_mode(self, mode:int):
        self.mode = mode

    def __len__(self):
        if self.mode == 0:
            return len(self.train_data)
        return len(self.valid_data)

    def __getitem__(self, idx):
        data = self.train_data
        if self.mode == 1:
            data = self.valid_data

        return data[idx]['x'], data[idx]['y']