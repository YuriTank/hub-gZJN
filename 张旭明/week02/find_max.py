"""
尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类
@author: zxm
@date: 2026/4/23
"""

from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from loguru import logger
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, in_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = in_features # 特征大小
        self.out_features = in_features # 输出分类数 = 输入维度大小
        self.model = nn.Sequential(
            nn.Linear(in_features, in_features),
        )

    def forward(self, x):
        y_prob = self.model.forward(x)
        return y_prob


class Trainer:
    """
    训练器 封装训练流程
    """
    def __init__(self, in_features,train_num,valid_num,test_num,epoch, batch,lr=1e-2):
        self.name = __file__
        # 训练、验证、测试数据集
        self.train_x,self.train_y = get_train_data(train_num,in_features)
        self.valid_x,self.valid_y = get_train_data(valid_num,in_features)
        self.test_x,self.test_y = get_train_data(test_num,in_features)
        # 模型
        self.model = TorchModel(in_features=in_features)
        # 优化器
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.epoch = epoch
        self.batch = batch
        self.criterion = CrossEntropyLoss()

        self.plot_data = defaultdict(list)
        self.init_logger()

    def init_logger(self):
        logger.add(f'{self.name}.log',encoding='utf8',level='DEBUG',mode='w')

    @staticmethod
    def iter_data_set(x, y, batch):
        """
        按照batch迭代
        """
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

    def _get_perclass_acc(self,y_predict,y_label):
        """
        计算每一类的准确率
        :param y_predict:
        :param y_label:
        :return:
        """
        num_class = self.model.out_features
        correct_predict = {}
        correct_total = {}
        for i in range(num_class):
            correct_mask = y_label == i  # 真实样本==i 的布尔标记（掩码矩阵）
            _correct_predict = (y_predict[correct_mask] == i).sum().item() # 预测i类且正确的样本数
            _correct_total = (y_label[correct_mask] == i).sum().item() # 真实i类样本

            correct_total[i] = _correct_total
            correct_predict[i] = _correct_predict

        logger.debug(f'[总样本数]:{y_label.shape[0]}')
        acc_list = []
        for i in range(num_class):
            pacc = correct_predict[i]/correct_total[i]+1e-8
            acc_list.append(pacc)
            logger.debug(f'[类别]: {i} [样本数]:{correct_total[i]} [预测正确数]:{correct_predict[i]} [准确率]: {pacc:.2f}')
        logger.debug(f'[平均准确率]:{np.mean(acc_list):.2f}')

    def run(self):
        self.train()
        self.test()
        self.save_model()
        self.plot()

    def save_model(self):
        name = f'{self.name}.bin'
        logger.debug(f'保存模型:{name}')
        torch.save(self.model,name)

    def train_epoch(self):
        """单个epoch，训练"""
        model = self.model
        model.train()
        total_loss = []
        total_acc = []
        for x, y in self.iter_data_set(self.train_x, self.train_y, self.batch):
            y_prob = model(x)  # batch,out
            y_predict = torch.argmax(y_prob, dim=1)  # batch,1
            loss = self.criterion(y_prob, y)
            # 反向传播 梯度更新并清零
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss.append(loss.item())
            acc = self._get_acc(y_predict, y)
            total_acc.append(acc)

        return np.mean(total_loss),np.mean(total_acc)

    def train(self):
        """
        迭代多个epoch，循环训练
        :return:
        """
        epoch = self.epoch
        logger.debug('开始训练')
        for e in range(epoch):
            # 训练
            train_loss,train_acc = self.train_epoch()
            self.plot_data['acc'].append(train_acc)
            self.plot_data['loss'].append(train_loss)
            # 验证
            _,_,eval_acc=self.eval(self.valid_x,self.valid_y)
            # 打印
            if e % 20 == 0:
                logger.debug(
                    f'[epoch]:{e} [loss]:{train_loss:.2f} [train_acc]:{train_acc:.2f} [valid_acc]:{eval_acc:.2f}')

        logger.debug('训练结束')

    def eval(self,x,y):
        """
        给定x，输出预测概率、预测标签、准确率
        :param x:
        :param y:
        :return:
        """
        y_prob, y_predict = self.predict(x)
        acc = self._get_acc(y_predict,y)
        return y_prob,y_predict,acc

    def test(self):
        """
        评估测试数据集
        :return:
        """
        logger.debug('开始评估')
        y_prob,y_predict,acc = self.eval(self.test_x,self.test_y)
        self._get_perclass_acc(y_predict,self.test_y)
        logger.debug('评估结束')

    def predict(self,x):
        """
        给定X，输出对应的Y，包含概率和分类
        :param x:
        :return:
        """
        model = self.model
        model.eval()
        with torch.no_grad():
            y_prob = model(x)  # batch,out
            y_predict = torch.argmax(y_prob, dim=1)  # batch,1
            return y_prob,y_predict

    def plot(self):
        """
        绘图，训练过程中的loss、acc
        :return:
        """
        epoch_steps = list(range(self.epoch))
        plt.plot(epoch_steps,self.plot_data['acc'],label='acc')
        plt.plot(epoch_steps,self.plot_data['loss'],label='loss')
        plt.legend()
        plt.savefig(f'{self.name}.png',dpi=300)
        plt.show()


def get_train_data(count, in_features):
    """
    返回训练数据集和标签，直接以张量大小返回
    :return:
    """
    x = np.random.rand(count, in_features)  # count,n
    y = np.argmax(x, axis=1)  # count  取最大数所在索引
    return torch.from_numpy(x).float(), torch.from_numpy(y).long()

def main():
    trainer = Trainer(in_features=NUM_IN_FEATURES,
                      train_num=NUM_TRAIN,valid_num=NUM_VALID,test_num=NUM_TEST,
                      epoch=NUM_EPOCH,batch=BATCH_SIZE)

    trainer.run()



if __name__ == '__main__':
    NUM_IN_FEATURES = 10 # 维度大小
    NUM_TRAIN = 1000
    NUM_VALID = 200
    NUM_TEST = 200
    NUM_EPOCH = 500
    BATCH_SIZE = 32
    main()
