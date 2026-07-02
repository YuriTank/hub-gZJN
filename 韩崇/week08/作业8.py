

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
WEEK8_ROOT = ROOT / "week8文本匹配问题" / "week8 文本匹配问题" / "文本匹配项目"
WEEK8_SRC = WEEK8_ROOT / "src"
if str(WEEK8_SRC) not in sys.path:
    sys.path.insert(0, str(WEEK8_SRC))

from dataset import PairDataset, CrossEncoderDataset
from dataset import build_pair_loaders, build_triplet_loader, build_crossencoder_loaders
from evaluate import eval_biencoder, eval_crossencoder
from model import build_biencoder, build_crossencoder
from train_biencoder import train_one_epoch_cosine, train_one_epoch_triplet
from train_crossencoder import train_one_epoch as train_one_epoch_crossencoder


DATA_ROOT = WEEK8_ROOT / "data"
BERT_PATH = ROOT / "week4语言模型" / "bert-base-chinese"
if not (BERT_PATH / "config.json").exists():
    BERT_PATH = WEEK8_ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_PATH = Path(__file__).resolve().with_name("文本匹配不同方法效果对比.md")
RESULT_PATH = LOG_DIR / "method_comparison.json"

DATASETS = ["lcqmc", "bq_corpus"]

METHODS = [
    {
        "key": "biencoder_cosine",
        "name": "BiEncoder + CosineEmbeddingLoss",
        "type": "biencoder",
        "loss": "cosine",
    },
    {
        "key": "biencoder_triplet",
        "name": "BiEncoder + TripletLoss",
        "type": "biencoder",
        "loss": "triplet",
    },
    {
        "key": "crossencoder",
        "name": "CrossEncoder + CrossEntropyLoss",
        "type": "crossencoder",
        "loss": None,
    },
]


# ── 工具函数 ──────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_method(key):
    for method in METHODS:
        if method["key"] == key:
            return method
    raise ValueError(f"未知方法：{key}")


def get_data_dir(dataset):
    return DATA_ROOT / dataset


def get_ckpt_path(dataset, method_key):
    return CKPT_DIR / f"{dataset}_{method_key}_best.pt"


def get_train_log_path(dataset, method_key):
    return LOG_DIR / f"{dataset}_{method_key}_log.json"


def to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value



def train_biencoder(dataset, method, args, device):
    data_dir = get_data_dir(dataset)
    ckpt_path = get_ckpt_path(dataset, method["key"])
    log_path = get_train_log_path(dataset, method["key"])

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    if method["loss"] == "cosine":
        train_loader, val_loader, _ = build_pair_loaders(
            data_dir, tokenizer,
            max_length=args.max_length, batch_size=args.batch_size,
        )
    else:
        train_loader, val_loader = build_triplet_loader(
            data_dir, tokenizer,
            max_length=args.max_length, batch_size=args.batch_size,
        )

    model = build_biencoder(
        bert_path=args.bert_path,
        pool=args.pool,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    optimizer = AdamW([
        {"params": list(model.bert.parameters()), "lr": args.lr},
        {"params": list(model.dropout.parameters()), "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    log_records = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if method["loss"] == "cosine":
            train_loss = train_one_epoch_cosine(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )
        else:
            train_loss = train_one_epoch_triplet(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum,
            )

        val_metrics = eval_biencoder(model, val_loader, device)
        elapsed = time.time() - t0
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        val_thr = val_metrics["threshold"]

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} threshold={val_thr:.2f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "dataset": dataset,
            "method": method["key"],
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "threshold": val_thr,
            "elapsed_s": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "threshold": val_thr,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "args": vars(args),
                "dataset": dataset,
                "method": method["key"],
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2, default=to_jsonable)


# ── 训练 CrossEncoder ────────────────────────────────────────────────────

def train_crossencoder(dataset, method, args, device):
    data_dir = get_data_dir(dataset)
    ckpt_path = get_ckpt_path(dataset, method["key"])
    log_path = get_train_log_path(dataset, method["key"])

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, _ = build_crossencoder_loaders(
        data_dir, tokenizer,
        max_length=args.cross_max_length, batch_size=args.batch_size,
    )

    model = build_crossencoder(
        bert_path=args.bert_path,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    optimizer = AdamW([
        {"params": list(model.bert.parameters()), "lr": args.lr},
        {"params": list(model.dropout.parameters()) + list(model.classifier.parameters()),
         "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    log_records = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch_crossencoder(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum,
        )

        val_metrics = eval_crossencoder(model, val_loader, device)
        elapsed = time.time() - t0
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "dataset": dataset,
            "method": method["key"],
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "elapsed_s": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "args": vars(args),
                "dataset": dataset,
                "method": method["key"],
            }, ckpt_path)
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2, default=to_jsonable)


# ── 加载 checkpoint 并评估 ────────────────────────────────────────────────

def evaluate_one(dataset, method, args, device):
    ckpt_path = get_ckpt_path(dataset, method["key"])
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    tokenizer = BertTokenizer.from_pretrained(saved_args.get("bert_path", args.bert_path))
    data_path = get_data_dir(dataset) / f"{args.eval_split}.jsonl"

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=saved_args.get("bert_path", args.bert_path),
            pool=saved_args.get("pool", args.pool),
            num_hidden_layers=saved_args.get("num_hidden_layers", args.num_hidden_layers),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds = PairDataset(data_path, tokenizer, max_length=saved_args.get("max_length", args.max_length))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(
            model, loader, device,
            find_threshold=False,
            threshold=ckpt.get("threshold", 0.5),
        )
        extra = f"threshold={metrics['threshold']:.2f}"
    else:
        model = build_crossencoder(
            bert_path=saved_args.get("bert_path", args.bert_path),
            num_hidden_layers=saved_args.get("num_hidden_layers", args.num_hidden_layers),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds = CrossEncoderDataset(data_path, tokenizer, max_length=args.cross_max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)
        extra = "argmax"

    return {
        "dataset": dataset,
        "method": method["key"],
        "method_name": method["name"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "extra": extra,
        "eval_split": args.eval_split,
        "eval_size": len(ds),
        "ckpt": str(ckpt_path),
    }


# ── 报告输出 ──────────────────────────────────────────────────────────────

def write_report(results, args):
    rows = []
    for r in results:
        rows.append(
            f"| {r['dataset']} | {r['method_name']} | {r['accuracy']:.4f} | "
            f"{r['f1']:.4f} | {r['extra']} | `{r['ckpt']}` |"
        )

    conclusions = []
    for dataset in args.datasets:
        items = [r for r in results if r["dataset"] == dataset]
        if not items:
            continue
        best = max(items, key=lambda x: x["f1"])
        conclusions.append(
            f"- `{dataset}` 上 F1(weighted) 最高的是 `{best['method_name']}`，F1={best['f1']:.4f}。"
        )

    content = f"""# 文本匹配不同方法效果对比

## 实验设置

- 数据集：{", ".join(args.datasets)}
- 对比方法：{", ".join(get_method(k)["name"] for k in args.methods)}
- 评估集：`{args.eval_split}`
- BERT 路径：`{args.bert_path}`
- epoch：{args.epochs}
- batch size：{args.batch_size}
- BiEncoder max length：{args.max_length}
- CrossEncoder max length：{args.cross_max_length}
- BERT 层数：{args.num_hidden_layers}
- 池化策略：{args.pool}

## 结果对比

| 数据集 | 方法 | Accuracy | F1(weighted) | 额外信息 | checkpoint |
|---|---|---:|---:|---|---|
{chr(10).join(rows)}

## 结论

{chr(10).join(conclusions)}
- BiEncoder 复用 week8 的阈值搜索逻辑，测试集评估时使用验证集保存的最优阈值。
- CrossEncoder 复用 week8 的句对拼接分类逻辑，直接用 argmax 得到预测类别。
"""
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write(content)


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="作业8：两个文本匹配数据集上的方法对比")
    parser.add_argument("--data_root", default=str(DATA_ROOT), type=str)
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--methods", nargs="+", default=[m["key"] for m in METHODS],
                        choices=[m["key"] for m in METHODS])
    parser.add_argument("--eval_split", default="test", choices=["validation", "test"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--report_path", default=str(REPORT_PATH), type=str)
    parser.add_argument("--result_path", default=str(RESULT_PATH), type=str)

    parser.add_argument("--pool", default="mean", choices=["cls", "mean", "max"])
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--cross_max_length", default=128, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--head_lr_mult", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("--margin", default=0.3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    # 命令行可覆盖 data_root，因此这里同步全局 DATA_ROOT
    global DATA_ROOT
    DATA_ROOT = Path(args.data_root)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"BERT 路径: {args.bert_path}")

    if not args.eval_only:
        for dataset in args.datasets:
            for method_key in args.methods:
                method = get_method(method_key)
                print(f"\n{'='*65}")
                print(f"训练 {dataset} / {method['name']}")
                if method["type"] == "biencoder":
                    train_biencoder(dataset, method, args, device)
                else:
                    train_crossencoder(dataset, method, args, device)

    all_results = []
    for dataset in args.datasets:
        for method_key in args.methods:
            method = get_method(method_key)
            print(f"\n{'='*65}")
            print(f"评估 {dataset} / {method['name']}")
            result = evaluate_one(dataset, method, args, device)
            if result is None:
                continue
            all_results.append(result)
            print(f"{result['method']:<20} "
                  f"Accuracy={result['accuracy']:.4f} "
                  f"F1(weighted)={result['f1']:.4f} "
                  f"{result['extra']}")

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=to_jsonable)
    write_report(all_results, args)

    print(f"\n对比日志 → {args.result_path}")
    print(f"实验报告 → {args.report_path}")


if __name__ == "__main__":
    main()
