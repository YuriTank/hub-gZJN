"""
BERT NER 训练脚本

教学重点：
  1. --use_crf 参数：一套脚本同时支持两种模型
  2. 分层学习率：BERT 层用 2e-5，分类头用 1e-4（加速头部收敛）
  3. Linear Warmup：防止训练初期大梯度破坏预训练参数
  4. seqeval 评估：entity-level F1（不是 token-level accuracy）

使用方式：
  python train.py                        # 训练 BERT+Linear（基线）
  python train.py --use_crf              # 训练 BERT+CRF
  python train.py --epochs 5 --lr 3e-5  # 自定义超参数

依赖：
  pip install torch transformers seqeval pytorch-crf tqdm
  export DASHSCOPE_API_KEY="sk-xxx"   （LLM对比时用）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import build_label_schema, build_dataloaders
from model import build_model

ROOT = Path(__file__).parent.parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[float, float]:
    """    在验证集上评估模型性能

    Args:
        model: 待评估的NER模型
        loader: 验证集DataLoader
        id2label: ID到标签的映射字典
        device: 计算设备(CPU/GPU)
        use_crf: 是否使用CRF层

    Returns:
        tuple[float, float]: (平均损失, 实体级别F1分数)

    功能说明:
        1. 遍历验证集所有批次，计算平均损失
        2. 收集所有预测结果和真实标签
        3. 使用seqeval计算实体级别的F1分数（比token级别更严格）
        4. CRF模型使用viterbi解码，非CRF模型使用argmax
    """
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    # 禁用梯度计算，节省显存和加速推理过程
    with torch.no_grad():
        # 遍历验证集的每个batch
        for batch in loader:
            # 将输入token IDs移到指定设备（GPU/CPU）
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)   # 将注意力掩码移到指定设备
            token_type_ids = batch["token_type_ids"].to(device) # 将token类型IDs移到指定设备（用于区分句子对）
            # 将标签移到指定设备
            labels = batch["labels"].to(device)

            # CRF模型和非CRF模型的前向传播及预测方式不同
            if use_crf:
                # CRF模式：获取发射分数和损失，使用viterbi解码最优路径
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                # 线性分类头模式：获取logits和损失，使用argmax取概率最大的类别
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            # 累加当前batch的损失值
            total_loss += loss.item()

            # 将标签从GPU移到CPU并转换为Python列表
            labels_np = labels.cpu().tolist()
            # 遍历batch中的每个样本
            for i in range(len(input_ids)):
                # 初始化真实标签序列列表
                gold_seq = []
                # 初始化预测标签序列列表
                pred_seq = []
                # 获取当前样本的标签序列
                token_labels = labels_np[i]
                # 这里的条件分支实际上两个分支代码相同，可以简化
                if use_crf:
                    # 获取CRF解码后的预测ID序列
                    pred_ids = pred_ids_list[i]
                else:
                    # 获取argmax后的预测ID序列
                    pred_ids = pred_ids_list[i]

                # 遍历当前样本的每个token位置
                for j, gold_id in enumerate(token_labels):
                    # 跳过padding位置的标签（-100是ignore_index）
                    if gold_id == -100:
                        continue
                    # 根据ID查找真实标签名称，添加到真实序列
                    gold_seq.append(id2label[gold_id])
                    # 根据是否使用CRF分别处理预测序列
                    if use_crf:
                        # 确保预测ID索引不越界
                        if j < len(pred_ids):
                            # 根据预测ID查找标签名称，找不到则默认为"O"
                            pred_seq.append(id2label.get(pred_ids[j], "O"))
                        else:
                            # 越界部分填充为"O"（非实体）
                            pred_seq.append("O")
                    else:
                        # 线性模式：直接根据预测ID查找标签名称
                        pred_seq.append(id2label.get(pred_ids[j], "O"))

                # 将当前样本的真实标签序列添加到所有真实序列列表
                all_golds.append(gold_seq)
                # 将当前样本的预测标签序列添加到所有预测序列列表
                all_preds.append(pred_seq)

    # 计算验证集的平均损失
    avg_loss = total_loss / len(loader)
    # 使用seqeval库计算实体级别的F1分数（基于精确匹配，比token级别更严格）
    entity_f1 = seqeval_f1(all_golds, all_preds)
    # 返回平均损失和实体F1分数
    return avg_loss, entity_f1


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)

        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 处理最后不足 grad_accum 的批次
    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def main():
    """
       主训练流程： orchestrates 完整的训练过程

       执行步骤:
           1. 解析命令行参数
           2. 初始化设备(GPU/CPU)
           3. 构建BIO标签体系和映射关系
           4. 加载预训练tokenizer并创建DataLoader
           5. 构建模型(BERT+Linear或BERT+CRF)
           6. 配置分层学习率优化器：
              - BERT层: 基础学习率(2e-5)，微调预训练参数
              - 分类头: 5倍学习率(1e-4)，加速新层收敛
           7. 配置Linear Warmup学习率调度器
           8. 循环训练指定epochs：
              - 训练一个epoch
              - 验证集评估
              - 保存最优模型(基于F1分数)
              - 记录训练日志
           9. 保存训练历史和模型checkpoint

       输出文件:
           - outputs/checkpoints/best_{crf/linear}.pt: 最优模型权重
           - outputs/logs/train_{crf/linear}.json: 训练曲线数据
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    # DataLoader
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
    )

    # 模型
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)

    # 分层学习率：BERT 层用基础 lr，分类头用 head_lr_mult 倍
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters()) +
        list(model.dropout.parameters()) +
        (list(model.crf.parameters()) if args.use_crf else [])
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")

    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = 0.0
    log_records = []

    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")
    print(f"\n下一步：python evaluate.py {'--use_crf' if args.use_crf else ''}")


def parse_args():
    """
    解析命令行参数
    使用示例:
        python train.py --use_crf --epochs 5 --lr 3e-5
    """
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
