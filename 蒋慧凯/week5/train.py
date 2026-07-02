#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 GPT 自回归语言模型
@author: jianghuikai
@date: 2026/05/21
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from loguru import logger
import matplotlib.pyplot as plt

from tokenizer import CharTokenizer
from dataset import CORPUS_PATH, CorpusDataset, collate_fn
from model import GPTModel
from functools import partial


def generate_sample(model, tokenizer, prompt="你好", max_new_tokens=30):
    """给定提示语，自回归生成一段文本"""
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id
        )
    output_text = tokenizer.decode(output_ids[0].tolist())
    return output_text


def main():
    logger.add("week5_train.log", encoding="utf-8", level="DEBUG")
    # 1. 超参数配置
    batch_size = 64
    epochs = 50
    learning_rate = 1e-3
    nlayers = 1
    d_model = 128
    max_seq_len = 64
    vocab_size = None
    # 2. 加载 tokenizer、构建 Dataset/DataLoader（划分 train/val）
    char_tokenizer = CharTokenizer(corpus_path=CORPUS_PATH)
    vocab_size = char_tokenizer.vocab_size
    corpus_dataset = CorpusDataset(CORPUS_PATH, char_tokenizer, max_seq_len)
    # 划分 train / val（比如 90% / 10%）
    train_size = int(0.9 * len(corpus_dataset))
    val_size = len(corpus_dataset) - train_size
    train_set, val_set = random_split(corpus_dataset, [train_size, val_size])
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=char_tokenizer.pad_id),
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=char_tokenizer.pad_id),
    )
    # 3. 初始化模型、优化器、损失函数
    gpt_model = GPTModel(
        d_model=d_model,
        n_layers=nlayers,
        vocab_size=char_tokenizer.vocab_size,
        max_seq_len=max_seq_len,
        n_head=8,
    )
    optimizer = Adam(gpt_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=char_tokenizer.pad_id)  # 忽略pad
    # 4. 训练循环（for epoch: train → eval → 保存最优 → 打印日志）

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        # 训练
        gpt_model.train()
        train_loss = 0
        for tokens, padding_mask in train_loader:
            optimizer.zero_grad()

            x = tokens[:, :-1]  # batch,seq
            y = tokens[:, 1:]  # batch,seq
            padding_mask = padding_mask[:, :-1]  # 和x保持一致
            predict_logits = gpt_model(x, padding_mask)  # batch,seq,vocab_size
            loss = criterion(predict_logits.view(-1, vocab_size), y.reshape(-1))
            loss.backward()
            train_loss += loss.item()

            # Transformer 训练加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), max_norm=1.0)
            optimizer.step()

        # 验证
        gpt_model.eval()
        val_loss = 0
        with torch.no_grad():
            for tokens, padding_mask in val_loader:
                x = tokens[:, :-1]  # batch,seq
                y = tokens[:, 1:]  # batch,seq
                padding_mask = padding_mask[:, :-1]  # 和x保持一致
                logits = gpt_model(x, padding_mask)
                loss = criterion(logits.view(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item()

        # 5. 绘制 loss 曲线
        # 计算平均 loss
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        logger.info(
            f"Epoch {epoch}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}"
        )

        # 保存最优模型（按 val_loss）
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(gpt_model.state_dict(), "week5_best_model.pth")

        # 快速验证：每轮用固定 prompt 生成一句，观察训练效果
        sample_text = generate_sample(
            gpt_model, char_tokenizer, prompt="你好", max_new_tokens=20
        )
        logger.info(f"[Epoch {epoch} Generate] {sample_text}")

    # 画图
    # 训练结束后画图保存
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("week5_train_loss.png", dpi=300)

    # 训练结束后：交互式生成测试
    logger.info("训练完成，进入交互式生成测试模式")
    print("\n===== 生成测试 =====")
    while True:
        try:
            prompt = input("请输入提示语（或按 Ctrl+C 退出）: ").strip()
            if not prompt:
                continue
            result = generate_sample(gpt_model, char_tokenizer, prompt=prompt)
            print(f"生成结果: {result}\n")
        except KeyboardInterrupt:
            print("\n退出生成测试")
            break


if __name__ == "__main__":
    main()
