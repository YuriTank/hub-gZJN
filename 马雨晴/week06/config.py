# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "class_num":2,
    "train_data_path": "D:/AI大模型课程/课程- 第七周 文本分类/week07_evaluate/data/train_tag_news.json",
    "valid_data_path": "D:/AI大模型课程/课程- 第七周 文本分类/week07_evaluate/data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 60,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path":r"C:\Users\11513\Desktop\bert-base-chinese",
    "seed": 987
}
