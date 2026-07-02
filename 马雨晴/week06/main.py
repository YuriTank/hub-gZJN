# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def analyse_data(config):
    """
    数据分析：正负样本数，文本平均长度等
    """
    train_data = load_data(config["train_data_path"], config)
    valid_data = load_data(config["valid_data_path"], config)
    train_data_positive_count = 0
    train_data_negative_count = 0
    train_data_text_lengths = []
    for data in train_data:
        input_ids, labels = data
        train_data_positive_count += (labels == 1).sum().item()
        train_data_negative_count += (labels == 0).sum().item()
        batch_lengths = (input_ids != 0).sum(dim=1).tolist()
        train_data_text_lengths.extend(batch_lengths)
    avg_length = np.mean(train_data_text_lengths)
    print(f"训练数据中正样本数量为：{train_data_positive_count},负样本数量为：{train_data_negative_count},文本平均长度: {avg_length:.2f}")
    valid_data_positive_count = 0
    valid_data_negative_count = 0
    valid_data_text_lengths = []
    for data in valid_data:
        input_ids, labels = data
        valid_data_positive_count += (labels == 1).sum().item()
        valid_data_negative_count += (labels == 0).sum().item()
        batch_lengths = (input_ids != 0).sum(dim=1).tolist()
        valid_data_text_lengths.extend(batch_lengths)
    avg_length = np.mean(valid_data_text_lengths)
    print(f"验证数据中正样本数量为：{valid_data_positive_count},负样本数量为：{valid_data_negative_count},文本平均长度: {avg_length:.2f}")
    
    
def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        # 在 main.py 的训练循环结束后，添加这段代码
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    analyse_data(Config)
    # 精简版：每个模型只跑最优参数组合
    best_results = {}

    # 基于经验的最优参数配置
    optimal_configs = {
        'gated_cnn': {
            'lr': 1e-3,
            'hidden_size': 256,
            'batch_size': 128,
            'pooling_style': 'avg'
        },
        'bert': {
            'lr': 1e-5,  # BERT需要更小的学习率
            'hidden_size': 768,
            'batch_size': 64,
            'pooling_style': 'avg'
        },
        'lstm': {
            'lr': 1e-3,
            'hidden_size': 256,
            'batch_size': 128,
            'pooling_style': 'avg'
        }
    }

    for model in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model
        
        # 使用最优配置
        opt = optimal_configs[model]
        Config["learning_rate"] = opt['lr']
        Config["hidden_size"] = opt['hidden_size']
        Config["batch_size"] = opt['batch_size']
        Config["pooling_style"] = opt['pooling_style']
        
        print(f"\n{'='*60}")
        print(f"训练 {model} 模型")
        print(f"配置: lr={opt['lr']}, hidden={opt['hidden_size']}, batch={opt['batch_size']}, pooling={opt['pooling_style']}")
        print(f"{'='*60}")
        
        acc = main(Config)
        
        best_results[model] = {
            'lr': opt['lr'],
            'hidden_size': opt['hidden_size'],
            'batch_size': opt['batch_size'],
            'pooling_style': opt['pooling_style'],
            'acc': acc
        }
        
        print(f"\n【{model}】准确率：{acc:.4f}")

    # 输出最终结果对比
    print("\n" + "="*60)
    print("三个模型结果对比")
    print("="*60)
    print(f"{'模型':<12} {'准确率':<10} {'学习率':<10} {'隐藏层':<10} {'批次':<8} {'池化方式':<10}")
    print("-"*60)
    for model, result in best_results.items():
        print(f"{model:<12} {result['acc']:.4f}     {result['lr']:<10} {result['hidden_size']:<10} {result['batch_size']:<8} {result['pooling_style']:<10}")

    # 找出整体最佳模型
    best_model = max(best_results, key=lambda x: best_results[x]['acc'])
    print("-"*60)
    print(f"\n 整体最佳模型：{best_model}，准确率：{best_results[best_model]['acc']:.4f}")
    print("="*60)
