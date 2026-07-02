#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
利用rnn预测特定字符在句子中的位置  特殊字符可配置
@author: hollis
@date: 2026/3/24 18:57
@description: 
"""
import os.path
import random
import string
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from loguru import logger
import matplotlib.pyplot as plt

SENTENCE_LENGTH = 20
SENTENCE_NUM = 1000
TAG_CHAR = '慧'  # 特殊字符，需要预测位置


class TorchModel(nn.Module):
    def __init__(self, in_features, out_features,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出分类数，句子长度+1，预测特定字符的位置
        self.embedding = nn.Embedding(in_features, 32)
        self.rnn = nn.RNN(32, 32, batch_first=True)
        self.avg_pool = nn.AvgPool1d(SENTENCE_LENGTH)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(32, self.out_features)

    def forward(self, x):
        x = self.embedding(x)  #  batch * sentence_len * 32
        output, hn = self.rnn(x)  # output: batch * sentence_len * 32, hn: batch * 32
        x = self.avg_pool(output.transpose(1, 2)).squeeze()  # batch * 32
        x = self.dropout(x)  # batch * 32
        y_prob = self.classifier(x)  # batch * sentence_len
        return y_prob


class Trainer:
    def __init__(self, train_num, valid_num, test_num, epoch, batch, lr=1e-2):
        self.name = __file__
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # 训练、验证、测试数据集
        self.train_vocab = Vocab(sentence_length=SENTENCE_LENGTH, sentence_count=train_num)
        self.valid_vocab = Vocab(sentence_length=SENTENCE_LENGTH, sentence_count=valid_num)
        self.test_vocab = Vocab(sentence_length=SENTENCE_LENGTH, sentence_count=test_num)
        self.train_x, self.train_y = get_train_data(self.train_vocab)
        self.valid_x, self.valid_y = get_train_data(self.valid_vocab)
        self.test_x, self.test_y = get_train_data(self.test_vocab)
        # 模型
        self.model = TorchModel(in_features=Vocab.vocab_size, out_features=SENTENCE_LENGTH + 1)  # 考虑不存在的情况 + 句子长度
        self.model_name = os.path.basename(__file__) +'.pth'
        # 优化器
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        # 损失函数
        self.criterion = CrossEntropyLoss()
        self.epoch = epoch
        self.batch = batch
        # 绘图
        self.plot_data = defaultdict(list)
        # 日志
        self.init_logger()

    def init_logger(self):
        # logger.remove()
        # logger.add(sys.stdout,level='DEBUG',format="{time} | {level} | {message}")
        log_path = os.path.join(self.script_dir, os.path.basename(self.name) + '.log')
        logger.add(log_path, encoding='utf8', level='DEBUG', mode='w')

    @staticmethod
    def iter_data_set(x, y, batch):
        # 按照batch迭代
        length = len(x)
        for i in range(0, length, batch):
            yield x[i:i + batch], y[i:i + batch]

    def _get_acc(self, y_predict, y_label):
        """
        计算预测、真实样本的准确率
        :param y_predict:
        :param y_label:
        :return:
        """
        acc = (y_predict == y_label).sum().item()
        batch = y_label.shape[0]
        acc = round(acc / batch, 2)
        return acc

    def run(self):
        self.train()
        self.test()
        self.plot()

    def save_model(self):
        # logger.debug(f'保存模型:{self.model_name}')
        model_path = os.path.join(self.script_dir, self.model_name)
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        # logger.debug(f'加载模型:{self.model_name}')
        model_path = os.path.join(self.script_dir, self.model_name)
        self.model.load_state_dict(torch.load(model_path))

    def train_epoch(self):
        """单个epoch，训练"""
        model = self.model
        model.train()
        total_loss = []
        total_acc = []
        for x, y in self.iter_data_set(self.train_x, self.train_y, self.batch):
            y_prob = model(x)  # batch,sentence_len
            y_predict = torch.argmax(y_prob, dim=1)  # batch,1
            loss = self.criterion(y_prob, y)
            # 反向传播 梯度更新并清零
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # 记录指标
            total_loss.append(loss.item())
            acc = self._get_acc(y_predict, y)
            total_acc.append(acc)

        return np.mean(total_loss), np.mean(total_acc)

    def train(self):
        """
        迭代多个epoch，循环训练
        :return:
        """
        epoch = self.epoch
        logger.debug('开始训练')
        best_acc = 0
        for e in range(epoch):
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.plot_data['acc'].append(train_acc)
            self.plot_data['loss'].append(train_loss)
            # 验证
            _, _, eval_acc = self.eval(self.valid_x, self.valid_y)
            # 保存
            if eval_acc > best_acc:
                best_acc = eval_acc
                self.save_model()
            # 打印
            if e % 20 == 0:
                logger.debug(
                    f'[epoch]:{e} [loss]:{train_loss:.2f} [train_acc]:{train_acc:.2f} [valid_acc]:{eval_acc:.2f}')

        logger.debug('训练结束')

    def eval(self, x, y):
        """
        给定x，输出预测概率、预测标签、准确率
        :param x:
        :param y:
        :return:
        """
        y_prob, y_predict = self.predict(x)
        acc = self._get_acc(y_predict, y)
        return y_prob, y_predict, acc

    def test(self):
        """
        评估测试数据集
        :return:
        """
        logger.debug('开始评估')
        self.load_model()
        y_prob, y_predict, acc = self.eval(self.test_x, self.test_y)
        logger.debug(f'[测试集] 准确率:{acc}')

        # 采样部分数据
        num_sample = 10
        logger.debug(f'[样例展示({num_sample})]')
        data_set = self.test_x[:num_sample] # batch,seq
        _, y_predict = self.predict(data_set)  # y_predict: batch

        # 转为列表 用于展示
        sentence_idx_list = data_set.tolist() # batch,seq
        y_predict = y_predict.tolist() # batch

        for sentence_idx,label in zip(sentence_idx_list,y_predict):
            sentence = self.test_vocab.map_idx2sentence(sentence_idx)
            char_label = sentence[label-1] if label > 0 else None
            is_true = char_label == TAG_CHAR
            logger.debug(f'{sentence} [是否含有特殊字符]:{TAG_CHAR in sentence} [预测位置]:{label} [对应字符]:{char_label}  [是否正确]:{is_true}')
            logger.debug('-'*10)

        logger.debug('评估结束')

    def predict(self, x)->Tuple[torch.Tensor,torch.Tensor]:
        """
        给定X，输出对应的Y，包含概率和分类
        :param x:
        :return:
        """
        model = self.model
        model.eval()
        with torch.no_grad():
            y_prob = model(x)  # batch,n
            y_predict = torch.argmax(y_prob, dim=1)  # batch,1
            return y_prob, y_predict

    def plot(self):
        """
        绘图，训练过程中的loss、acc
        :return:
        """
        epoch_steps = list(range(self.epoch))
        plt.plot(epoch_steps, self.plot_data['acc'], label='acc')
        plt.plot(epoch_steps, self.plot_data['loss'], label='loss')
        plt.legend()
        plot_path = os.path.join(self.script_dir, os.path.basename(self.name) + '.png')
        plt.savefig(plot_path, dpi=300)
        plt.show()


class Vocab:
    UNK_CHAR = '<UNK>'
    PAD_CHAR = '<PAD>'
    alpha_num_letters = list(string.ascii_letters + string.digits)  # 字母、数字
    letters = alpha_num_letters + [UNK_CHAR, PAD_CHAR, TAG_CHAR] # 所有字母集合
    vocab: Dict[str, int] = {}

    def __init__(self, sentence_length: int, sentence_count: int):
        self.sentence_count = sentence_count
        self.sentence_length = sentence_length
        self.sentence_list: List[str] = []

        self.build_vocab()
        self.build_sentence_data_set(self.sentence_count)

    @classmethod
    @property
    def vocab_size(cls) -> int:
        return len(cls.vocab)

    def build_sentence(self) -> str:
        """
        按照0.5的概率随机生成含有特定字符的句子
        :return:
        """
        if random.random() < 0.5:
            # 插入特殊字符
            chars = self._combine_letters(total_length=self.sentence_length - 1)
            chars.append(TAG_CHAR)
        else:
            # 不插入
            chars = self._combine_letters(total_length=self.sentence_length)
        # 随机打乱字符
        random.shuffle(chars)
        return ''.join(chars)

    def _combine_letters(self, total_length: int) -> List[str]:
        return list(random.choices(self.alpha_num_letters, k=total_length))

    def build_sentence_data_set(self, count: int):
        """
        构造指定大小的句子集合
        :param count:
        :return:
        """
        for _ in range(count):
            self.sentence_list.append(self.build_sentence())

    @classmethod
    def build_vocab(cls):
        """
        构建词典,PAD:0, UNK 末位
        :return:
        """
        if cls.vocab:
            return

        c2idx_mapping= {c:idx for idx,c in enumerate(cls.letters)}
        cls.vocab = c2idx_mapping

    def get_sentence_data_set(self) -> List[str]:
        return self.sentence_list

    def map_sentence2idx(self, sentence: str) -> List:
        """
        获取句子对应的索引编号
        :param sentence:
        :return:
        """
        idx = []
        unk_idx = self.vocab[self.UNK_CHAR]
        for c in sentence:
            idx.append(self.vocab.get(c, unk_idx))
        return idx

    def map_idx2sentence(self,idx_list:List[int]):
        """获取编号对应的字符，拼接成句子"""
        chars = [self.letters[idx] for idx in idx_list]
        return ''.join(chars)



def get_train_data(vocab: Vocab) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回训练数据集和标签，直接以张量大小返回
    :return:
    """
    sentence_list = vocab.get_sentence_data_set()
    x = []
    y = []
    for sentence in sentence_list:
        sentence_idx: List = vocab.map_sentence2idx(sentence)
        tag = 0  # 表示没有
        if TAG_CHAR in sentence:
            tag = sentence.index(TAG_CHAR) + 1  # 从序号1开始

        x.append(sentence_idx)
        y.append(tag)
    return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)



def main():
    trainer = Trainer(train_num=5000, valid_num=100, test_num=100,
                      epoch=300, batch=32, lr=1e-3)

    trainer.run()


if __name__ == '__main__':
    main()
