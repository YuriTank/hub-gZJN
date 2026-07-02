"""
文本匹配全流程整合代码
包含：数据探索、BiEncoder/CrossEncoder模型、数据集构建、训练、评估、对比分析、BadCase分析
执行顺序：
1. 数据探索 (explore_data)
2. 模型定义 (BiEncoder/CrossEncoder)
3. 数据集构建 (Pair/Triplet/CrossEncoderDataset)
4. 评估工具 (eval_biencoder/eval_crossencoder)
5. BiEncoder训练 (train_biencoder)
6. CrossEncoder训练 (train_crossencoder)
7. 多方法对比 (compare_methods)
8. BadCase分析 (analyze_badcases)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertConfig, BertModel, BertTokenizer,
    get_linear_schedule_with_warmup,
    logging as transformers_logging
)
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
    confusion_matrix
)

# ===================== 全局配置 =====================
random.seed(42)
torch.manual_seed(42)

# 默认路径（根据实际环境调整）
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "bq_corpus"
BERT_PATH = ROOT.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "chen_outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIG_DIR = OUTPUT_DIR / "figures"

# 创建目录
for dir_path in [OUTPUT_DIR, CKPT_DIR, LOG_DIR, FIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===================== 1. 数据探索模块 (explore_data) =====================
_CN_FONT = None
def _get_font():
    global _CN_FONT
    if _CN_FONT is None:
        try:
            font_path = next(
                p for p in fm.findSystemFonts()
                if any(k in p.lower() for k in ("simhei", "msyh", "simsun", "notosans"))
            )
            _CN_FONT = fm.FontProperties(fname=font_path)
        except StopIteration:
            _CN_FONT = fm.FontProperties()
    return _CN_FONT

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def plot_label_distribution(splits_data, output_dir):
    fig, axes = plt.subplots(1, len(splits_data), figsize=(10, 4))
    if len(splits_data) == 1:
        axes = [axes]

    fp = _get_font()
    for ax, (split_name, rows) in zip(axes, splits_data.items()):
        labels = [r["label"] for r in rows]
        cnt = Counter(labels)
        counts = [cnt.get(0, 0), cnt.get(1, 0)]
        bars = ax.bar(["不相似 (0)", "相似 (1)"], counts,
                      color=["#F44336", "#2196F3"], width=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{c}\n({c/len(rows)*100:.1f}%)", ha="center", va="bottom",
                    fontproperties=fp, fontsize=9)
        ax.set_title(f"{split_name}（{len(rows):,} 条）", fontproperties=fp)
        ax.set_ylabel("数量", fontproperties=fp)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("标签分布（正例约 31%，负例约 69%）", fontproperties=fp,
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "label_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  标签分布图表已保存 → {save_path}")

def plot_char_length(rows, output_dir):
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]

    def lens(rs):
        return [len(r["sentence1"]) for r in rs] + [len(r["sentence2"]) for r in rs]

    pos_lens = lens(pos_rows)
    neg_lens = lens(neg_rows)
    all_lens = pos_lens + neg_lens

    max_len = np.percentile(all_lens, 99)
    if max_len < 100:
        max_len = 100

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_lens, bins=40, alpha=0.6, label="正样本（相似）",
            color="#2196F3", density=True)
    ax.hist(neg_lens, bins=40, alpha=0.6, label="负样本（不相似）",
            color="#F44336", density=True)
    ax.axvline(32, color="black", linestyle="--", linewidth=1,
               label="max_length=32")
    ax.axvline(64, color="gray", linestyle="--", linewidth=1,
               label="max_length=64")
    ax.set_xlabel("句子字符长度", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度分布（train）", fontproperties=fp)
    ax.legend(prop=fp)
    ax.set_xlim(0, max_len)

    fig.tight_layout()
    save_path = output_dir / "char_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  字符长度图表已保存 → {save_path}")

    all_lens = [len(r["sentence1"]) for r in rows] + [len(r["sentence2"]) for r in rows]
    print(f"  字符长度统计（train 全部句子）：")
    print(f"    均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    for threshold in [32, 48, 64, 96]:
        cover = sum(1 for l in all_lens if l <= threshold) / len(all_lens) * 100
        print(f"    max_length={threshold:3d} 覆盖率: {cover:.1f}%")

def plot_token_length(rows, tokenizer, output_dir):
    print("  计算 Token 长度（需要 tokenize，稍慢...）")
    token_lens = []
    for r in rows[:5000]:
        t1 = len(tokenizer.tokenize(r["sentence1"]))
        t2 = len(tokenizer.tokenize(r["sentence2"]))
        token_lens.extend([t1, t2])

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_lens, bins=40, color="#4CAF50", alpha=0.8, density=True)
    ax.axvline(np.mean(token_lens), color="red", linestyle="-",
               label=f"均值={np.mean(token_lens):.1f}")
    ax.axvline(np.percentile(token_lens, 95), color="orange", linestyle="--",
               label=f"P95={np.percentile(token_lens, 95):.0f}")
    ax.set_xlabel("单句 Token 数（不含 [CLS]/[SEP]）", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("单句 Token 数分布（train 前 5000 条）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "token_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Token 长度图表已保存 → {save_path}")
    print(f"  Token 长度：均值={np.mean(token_lens):.1f}  "
          f"P95={np.percentile(token_lens, 95):.0f}  最长={max(token_lens)}")

def plot_length_diff(rows, output_dir):
    pos_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 1]
    neg_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 0]
    
    all_diffs = pos_diffs + neg_diffs
    max_diff = np.percentile(all_diffs, 99)
    if max_diff < 50:
        max_diff = 50

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_diffs, bins=30, alpha=0.6, label=f"正样本 均值={np.mean(pos_diffs):.1f}",
            color="#2196F3", density=True)
    ax.hist(neg_diffs, bins=30, alpha=0.6, label=f"负样本 均值={np.mean(neg_diffs):.1f}",
            color="#F44336", density=True)
    ax.set_xlabel("|len(s1) - len(s2)| 字符数", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度差分布（length bias 检测）", fontproperties=fp)
    ax.legend(prop=fp)
    ax.set_xlim(0, max_diff)
    
    fig.tight_layout()
    save_path = output_dir / "length_diff_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  长度差图表已保存 → {save_path}")
    print(f"  长度差：正样本均值={np.mean(pos_diffs):.1f}  负样本均值={np.mean(neg_diffs):.1f}")
    if np.mean(pos_diffs) < np.mean(neg_diffs) * 0.7:
        print("  ⚠️  正样本长度差明显更小，存在 length bias 风险")
    else:
        print("  ✓ 正/负样本长度差接近，无明显 length bias")

def print_stats(name, rows):
    labels = [r["label"] for r in rows]
    cnt = Counter(labels)
    s1_lens = [len(r["sentence1"]) for r in rows]
    s2_lens = [len(r["sentence2"]) for r in rows]
    all_lens = s1_lens + s2_lens

    print(f"\n{'='*50}")
    print(f"【{name}】共 {len(rows):,} 条")
    print(f"{'='*50}")

    n_pos = cnt.get(1, 0)
    n_neg = cnt.get(0, 0)
    n_unlabeled = sum(v for k, v in cnt.items() if k not in (0, 1))
    if n_unlabeled:
        print(f"  标签未公开（CLUE 竞赛格式）: {n_unlabeled:>6,} 条  —— 仅供参考，不用于评估")
    else:
        print(f"  正样本（相似）  : {n_pos:>6,} ({n_pos/len(rows)*100:.1f}%)")
        print(f"  负样本（不相似）: {n_neg:>6,} ({n_neg/len(rows)*100:.1f}%)")
        print(f"  不均衡比 (neg/pos): {n_neg/max(n_pos, 1):.1f}x")
    print(f"  句子字符长度 — 均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    print(f"  示例正样本：")
    for r in [r for r in rows if r["label"] == 1][:2]:
        print(f"    ✓  {r['sentence1']!r}  ||  {r['sentence2']!r}")
    print(f"  示例负样本：")
    for r in [r for r in rows if r["label"] == 0][:2]:
        print(f"    ✗  {r['sentence1']!r}  ||  {r['sentence2']!r}")

def explore_data(args):
    splits = {}
    for split in ["train", "validation", "test"]:
        path = args.data_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = load_jsonl(path)

    for name, rows in splits.items():
        print_stats(name, rows)

    train_rows = splits.get("train", [])
    if not train_rows:
        print("train.jsonl 不存在，请先准备数据集！")
        return

    print(f"\n{'='*50}")
    print("生成可视化图表...")

    plot_label_distribution(splits, args.output_dir)
    plot_char_length(train_rows, args.output_dir)
    plot_length_diff(train_rows, args.output_dir)

    if not args.skip_token:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        plot_token_length(train_rows, tokenizer, args.output_dir)

    print(f"\n所有图表已保存至 → {args.output_dir}")

# ===================== 2. 模型定义模块 (model) =====================
class BiEncoder(nn.Module):
    """表示型文本匹配：Siamese Bi-Encoder"""
    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
        super().__init__()
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers_logging.set_verbosity(_prev)

        self.pool = pool
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids, attention_mask, token_type_ids):
        """单句编码，返回 L2 归一化后的句向量 [B, H]"""
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        vec = self._pool(out.last_hidden_state, attention_mask)
        vec = self.dropout(vec)
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, batch_a, batch_b):
        """返回 (emb_a, emb_b)，各形状 [B, H]"""
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b

    def _pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()

        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values

class CrossEncoder(nn.Module):
    """交互型文本匹配：Cross-Encoder"""
    def __init__(self, bert_path, num_hidden_layers=None):
        super().__init__()
        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers_logging.set_verbosity(_prev)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = out.last_hidden_state[:, 0, :]
        cls_vec = self.dropout(cls_vec)
        logits = self.classifier(cls_vec)
        return logits

def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    return BiEncoder(bert_path, pool, dropout, num_hidden_layers)

def build_crossencoder(bert_path, num_hidden_layers=None):
    return CrossEncoder(bert_path, num_hidden_layers)

# ===================== 3. 数据集构建模块 (dataset) =====================
def encode_single(tokenizer, text, max_length):
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "token_type_ids": enc["token_type_ids"].squeeze(0),
    }

class PairDataset(Dataset):
    """句对数据集：(sentence1, sentence2, label)"""
    def __init__(self, data_path, tokenizer, max_length=64):
        self.rows = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc_a = encode_single(self.tokenizer, r["sentence1"], self.max_length)
        enc_b = encode_single(self.tokenizer, r["sentence2"], self.max_length)
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "token_type_ids_b": enc_b["token_type_ids"],
            "label": torch.tensor(r["label"], dtype=torch.long),
        }

class TripletDataset(Dataset):
    """三元组数据集：(anchor, positive, negative)"""
    def __init__(self, data_path, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = self._build_triplets(load_jsonl(data_path))

    def _build_triplets(self, rows):
        neg_by_sent = defaultdict(list)
        pos_pairs = []
        all_sents = set()

        for r in rows:
            s1, s2, label = r["sentence1"], r["sentence2"], r["label"]
            all_sents.add(s1)
            all_sents.add(s2)
            if label == 0:
                neg_by_sent[s1].append(s2)
                neg_by_sent[s2].append(s1)
            else:
                pos_pairs.append((s1, s2))

        global_neg = list(all_sents)
        triplets = []
        for anchor, positive in pos_pairs:
            negs = neg_by_sent.get(anchor, [])
            if not negs:
                negs = [random.choice(global_neg) for _ in range(1)]
            negative = random.choice(negs)
            triplets.append((anchor, positive, negative))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        enc_a = encode_single(self.tokenizer, anchor, self.max_length)
        enc_p = encode_single(self.tokenizer, positive, self.max_length)
        enc_n = encode_single(self.tokenizer, negative, self.max_length)
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_p": enc_p["input_ids"],
            "attention_mask_p": enc_p["attention_mask"],
            "token_type_ids_p": enc_p["token_type_ids"],
            "input_ids_n": enc_n["input_ids"],
            "attention_mask_n": enc_n["attention_mask"],
            "token_type_ids_n": enc_n["token_type_ids"],
        }

class CrossEncoderDataset(Dataset):
    """CrossEncoder专用数据集：拼接句对为单序列"""
    def __init__(self, data_path, tokenizer, max_length=128):
        self.rows = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tokenizer(
            r["sentence1"],
            r["sentence2"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.long),
        }

def build_pair_loaders(data_dir, tokenizer, max_length=64, batch_size=32):
    train_ds = PairDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds = PairDataset(data_dir / "test.jsonl", tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

def build_triplet_loader(data_dir, tokenizer, max_length=64, batch_size=32):
    train_ds = TripletDataset(data_dir / "train.jsonl", tokenizer, max_length)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

def build_crossencoder_loaders(data_dir, tokenizer, max_length=128, batch_size=32):
    train_ds = CrossEncoderDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = CrossEncoderDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds = CrossEncoderDataset(data_dir / "test.jsonl", tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

# ===================== 4. 评估工具模块 (evaluate) =====================
def _find_best_threshold(sims, labels):
    """枚举阈值，返回使 weighted-F1 最高的阈值"""
    best_f1, best_thresh = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh)

@torch.no_grad()
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5):
    """BiEncoder评估：计算余弦相似度 + 阈值搜索"""
    model.eval()
    all_sims, all_labels = [], []

    for batch in loader:
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        all_sims.extend(sims)
        all_labels.extend(batch["label"].tolist())

    sims = np.array(all_sims)
    labels = np.array(all_labels)

    if find_threshold:
        threshold = _find_best_threshold(sims, labels)

    preds = (sims >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(labels, sims)
    except ValueError:
        auc = float("nan")

    return {
        "similarities": all_sims,
        "labels": all_labels,
        "accuracy": accuracy,
        "f1": f1,
        "threshold": threshold,
        "auc": auc,
    }

@torch.no_grad()
def eval_crossencoder(model, loader, device):
    """CrossEncoder评估：直接分类，无需阈值"""
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].tolist()

        logits = model(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        preds = logits.argmax(dim=-1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
    }

def plot_similarity_distribution(sims, labels, save_path, title, color):
    """绘制正负样本相似度分布"""
    fp = _get_font()
    pos_sims = [s for s, l in zip(sims, labels) if l == 1]
    neg_sims = [s for s, l in zip(sims, labels) if l == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_sims, bins=40, alpha=0.6, label="正样本", color=color, density=True)
    ax.hist(neg_sims, bins=40, alpha=0.6, label="负样本", color="#F44336", density=True)
    ax.set_xlabel("余弦相似度", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title(title, fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ===================== 5. BiEncoder训练模块 (train_biencoder) =====================
def train_one_epoch_cosine(model, loader, optimizer, scheduler, device,
                           epoch, total_epochs, margin, grad_accum):
    """CosineEmbeddingLoss训练"""
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Cosine]", leave=False)
    for step, batch in enumerate(pbar):
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        labels = batch["label"].to(device)

        emb_a, emb_b = model(batch_a, batch_b)
        cos_target = (labels.float() * 2 - 1)
        loss = F.cosine_embedding_loss(emb_a, emb_b, cos_target, margin=margin)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples

def train_one_epoch_triplet(model, loader, optimizer, scheduler, device,
                            epoch, total_epochs, margin, grad_accum):
    """TripletLoss训练"""
    model.train()
    total_loss, total_samples = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Triplet]", leave=False)
    for step, batch in enumerate(pbar):
        enc_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        enc_p = {
            "input_ids": batch["input_ids_p"].to(device),
            "attention_mask": batch["attention_mask_p"].to(device),
            "token_type_ids": batch["token_type_ids_p"].to(device),
        }
        enc_n = {
            "input_ids": batch["input_ids_n"].to(device),
            "attention_mask": batch["attention_mask_n"].to(device),
            "token_type_ids": batch["token_type_ids_n"].to(device),
        }

        emb_a = model.encode(**enc_a)
        emb_p = model.encode(**enc_p)
        emb_n = model.encode(**enc_n)

        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=margin)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * emb_a.size(0)
        total_samples += emb_a.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples

def train_biencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"Loss类型: {args.loss}  BERT层数: {args.num_hidden_layers}  Epochs: {args.epochs}")

    # 初始化Tokenizer和DataLoader
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    if args.loss == "cosine":
        train_loader, val_loader, _ = build_pair_loaders(
            args.data_dir, tokenizer, args.max_length, args.batch_size
        )
    else:
        train_loader = build_triplet_loader(
            args.data_dir, tokenizer, args.max_length, args.batch_size
        )
        val_loader, _, _ = build_pair_loaders(
            args.data_dir, tokenizer, args.max_length, args.batch_size
        )

    # 构建模型
    model = build_biencoder(
        bert_path=args.bert_path,
        pool=args.pool,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 训练主循环
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        if args.loss == "cosine":
            train_loss = train_one_epoch_cosine(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum
            )
        else:
            train_loss = train_one_epoch_triplet(
                model, train_loader, optimizer, scheduler, device,
                epoch, args.epochs, args.margin, args.grad_accum
            )

        # 验证
        metrics = eval_biencoder(model, val_loader, device)
        val_acc, val_f1 = metrics["accuracy"], metrics["f1"]
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # 保存最优模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = CKPT_DIR / f"biencoder_{args.loss}_best.pt"
            torch.save({
                "args": vars(args),
                "state_dict": model.state_dict(),
                "best_f1": best_f1,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  最优模型已保存 → {ckpt_path}")

    print(f"\n训练完成 | 最优Val F1: {best_f1:.4f}")

# ===================== 6. CrossEncoder训练模块 (train_crossencoder) =====================
def train_one_epoch_cross(model, loader, optimizer, scheduler, criterion,
                          device, epoch, total_epochs, grad_accum):
    """CrossEncoder单轮训练"""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [CrossEncoder]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples

def train_crossencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"BERT层数: {args.num_hidden_layers}  Epochs: {args.epochs}  Batch size: {args.batch_size}")

    # Tokenizer和DataLoader
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, _ = build_crossencoder_loaders(
        args.data_dir, tokenizer, args.max_length, args.batch_size
    )

    # 模型
    model = build_crossencoder(
        bert_path=args.bert_path,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 训练
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch_cross(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, args.grad_accum
        )

        # 验证
        metrics = eval_crossencoder(model, val_loader, device)
        val_acc, val_f1 = metrics["accuracy"], metrics["f1"]
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # 保存最优
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_path = CKPT_DIR / "crossencoder_best.pt"
            torch.save({
                "args": vars(args),
                "state_dict": model.state_dict(),
                "best_f1": best_f1,
                "epoch": epoch,
            }, ckpt_path)
            print(f"  最优模型已保存 → {ckpt_path}")

    print(f"\n训练完成 | 最优Val F1: {best_f1:.4f}")

# ===================== 7. 多方法对比模块 (compare_methods) =====================
METHODS = [
    {
        "key": "biencoder_cosine",
        "label": "BiEncoder\n(CosineEmbeddingLoss)",
        "ckpt": "biencoder_cosine_best.pt",
        "type": "biencoder",
        "color": "#2196F3",
    },
    {
        "key": "biencoder_triplet",
        "label": "BiEncoder\n(TripletLoss)",
        "ckpt": "biencoder_triplet_best.pt",
        "type": "biencoder",
        "color": "#4CAF50",
    },
    {
        "key": "crossencoder",
        "label": "CrossEncoder\n(CrossEntropyLoss)",
        "ckpt": "crossencoder_best.pt",
        "type": "crossencoder",
        "color": "#FF9800",
    },
]

def load_and_eval(method, tokenizer, device, split, batch_size):
    """加载模型并评估"""
    ckpt_path = CKPT_DIR / method["ckpt"]
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = DATA_DIR / f"{split}.jsonl"
        ds = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)

    else:
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = DATA_DIR / f"{split}.jsonl"
        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    metrics["model"] = model
    metrics["ckpt"] = ckpt
    return metrics

def plot_comparison_bar(results, save_path):
    """绘制准确率/F1对比柱状图"""
    names = [m["label"] for m in results]
    accs = [m["metrics"]["accuracy"] for m in results]
    f1s = [m["metrics"]["f1"] for m in results]
    colors = [m["color"] for m in results]

    fp = _get_font()
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, accs, width, label="准确率", color=colors, alpha=0.8)
    rects2 = ax.bar(x + width/2, f1s, width, label="F1", color=colors, alpha=0.5)

    ax.set_ylabel("分数", fontproperties=fp)
    ax.set_title("不同模型效果对比", fontproperties=fp)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontproperties=fp)
    ax.legend(prop=fp)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontproperties=fp, fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  对比柱状图已保存 → {save_path}")

def compare_methods(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # 加载并评估所有方法
    results = []
    for method in METHODS:
        print(f"\n评估 {method['label']}...")
        metrics = load_and_eval(method, tokenizer, device, args.split, args.batch_size)
        if metrics is None:
            continue
        results.append({
            "method": method,
            "metrics": metrics,
        })

    if not results:
        print("无可用模型结果！")
        return

    # 绘制对比图
    plot_comparison_bar(results, FIG_DIR / "method_comparison.png")

    # 输出详细指标
    print(f"\n{'='*60}")
    print("各方法详细指标（Validation集）")
    print(f"{'='*60}")
    for res in results:
        m = res["method"]
        metrics = res["metrics"]
        print(f"\n【{m['label']}】")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  F1:     {metrics['f1']:.4f}")
        if "auc" in metrics:
            print(f"  AUC:    {metrics['auc']:.4f}")
        if m["type"] == "biencoder":
            print(f"  最优阈值: {metrics['threshold']:.3f}")

# ===================== 8. BadCase分析模块 (analyze_badcases) =====================
@torch.no_grad()
def collect_biencoder_preds(model, loader, raw_rows, device, threshold):
    """收集BiEncoder预测结果"""
    model.eval()
    results = []
    idx = 0
    for batch in loader:
        batch_a = {
            "input_ids": batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids": batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        labels = batch["label"].tolist()

        for sim, label in zip(sims, labels):
            row = raw_rows[idx]
            results.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": label,
                "score": sim,
                "pred": int(sim >= threshold),
            })
            idx += 1
    return results

@torch.no_grad()
def collect_crossencoder_preds(model, loader, raw_rows, device):
    """收集CrossEncoder预测结果"""
    model.eval()
    results = []
    idx = 0
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["token_type_ids"].to(device),
        ).cpu()
        probs = F.softmax(logits, dim=-1)[:, 1].tolist()
        preds = logits.argmax(dim=-1).tolist()
        labels = batch["label"].tolist()

        for prob, pred, label in zip(probs, preds, labels):
            row = raw_rows[idx]
            results.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label": label,
                "score": prob,
                "pred": pred,
            })
            idx += 1
    return results

def split_badcases(results, threshold=0.5):
    """分类BadCase：FP/FN + 置信度分级"""
    fp_high = []  # 高置信度FP: pred=1, label=0, score>threshold
    fp_low = []   # 低置信度FP: pred=1, label=0, score<=threshold
    fn_high = []  # 高置信度FN: pred=0, label=1, score<threshold
    fn_low = []   # 低置信度FN: pred=0, label=1, score>=threshold

    for r in results:
        if r["pred"] == 1 and r["label"] == 0:
            if r["score"] > threshold:
                fp_high.append(r)
            else:
                fp_low.append(r)
        elif r["pred"] == 0 and r["label"] == 1:
            if r["score"] < threshold:
                fn_high.append(r)
            else:
                fn_low.append(r)

    return {
        "FP_high": fp_high,
        "FP_low": fp_low,
        "FN_high": fn_high,
        "FN_low": fn_low,
        "total_bad": len(fp_high) + len(fp_low) + len(fn_high) + len(fn_low),
    }

def print_badcases(badcases, n_cases=10):
    """打印BadCase示例"""
    print(f"\n{'='*60}")
    print(f"BadCase 统计（总计: {badcases['total_bad']} 条）")
    print(f"{'='*60}")
    print(f"高置信度FP（假阳性）: {len(badcases['FP_high'])} 条")
    print(f"低置信度FP（假阳性）: {len(badcases['FP_low'])} 条")
    print(f"高置信度FN（假阴性）: {len(badcases['FN_high'])} 条")
    print(f"低置信度FN（假阴性）: {len(badcases['FN_low'])} 条")

    # 打印示例
    for case_type, name in [
        ("FP_high", "高置信度FP（模型确信相似但实际不同）"),
        ("FN_high", "高置信度FN（模型确信不同但实际相似）"),
        ("FP_low", "低置信度FP（边界错误）"),
        ("FN_low", "低置信度FN（边界错误）"),
    ]:
        cases = badcases[case_type][:n_cases]
        if not cases:
            continue
        print(f"\n【{name}】（示例 {len(cases)} 条）")
        for i, r in enumerate(cases, 1):
            print(f"  {i}. 分数: {r['score']:.3f}")
            print(f"     S1: {r['sentence1']!r}")
            print(f"     S2: {r['sentence2']!r}")

def analyze_badcases(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # 加载模型
    ckpt_path = args.ckpt if args.ckpt else CKPT_DIR / f"{args.model_type}_best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint不存在: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    # 加载数据
    raw_rows = load_jsonl(DATA_DIR / "validation.jsonl")

    # 初始化模型和Loader
    if args.model_type == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        ds = PairDataset(DATA_DIR / "validation.jsonl", tokenizer, saved_args.get("max_length", 64))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        # 先评估获取最优阈值
        metrics = eval_biencoder(model, loader, device)
        preds = collect_biencoder_preds(model, loader, raw_rows, device, metrics["threshold"])
    else:
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        ds = CrossEncoderDataset(DATA_DIR / "validation.jsonl", tokenizer, saved_args.get("max_length", 128))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        preds = collect_crossencoder_preds(model, loader, raw_rows, device)

    # 分析BadCase
    badcases = split_badcases(preds)
    print_badcases(badcases, args.n_cases)

    # 输出优化建议
    print(f"\n{'='*60}")
    print("优化方向建议")
    print(f"{'='*60}")
    if len(badcases["FP_high"]) > len(badcases["FN_high"]):
        print("1. 假阳性偏多：模型对表面相似但语义不同的句子过度敏感")
        print("   → 优化：增加难负样本、引入对比学习、使用语义增强的预训练模型")
    else:
        print("1. 假阴性偏多：模型未能捕捉语义相似但表述不同的句子")
        print("   → 优化：数据增强（同义改写）、增加正样本多样性、使用更大的模型")

    if len(badcases["FP_low"]) + len(badcases["FN_low"]) > badcases["total_bad"] * 0.5:
        print("2. 边界错误偏多：模型对模糊样本的判别能力不足")
        print("   → 优化：增加边界样本、调整损失函数margin、延长训练时间")

# ===================== 主函数 & 命令行参数 =====================
def main():
    parser = argparse.ArgumentParser(description="文本匹配全流程工具")
    subparsers = parser.add_subparsers(dest="command", required=True, help="子命令")

    # 1. 数据探索
    parser_explore = subparsers.add_parser("explore", help="数据探索")
    parser_explore.add_argument("--data_dir", default=DATA_DIR, type=Path)
    parser_explore.add_argument("--bert_path", default=BERT_PATH, type=str)
    parser_explore.add_argument("--output_dir", default=FIG_DIR, type=Path)
    parser_explore.add_argument("--skip_token", action="store_true", help="跳过Token长度分析")

    # 2. 训练BiEncoder
    parser_train_bi = subparsers.add_parser("train_bi", help="训练BiEncoder")
    parser_train_bi.add_argument("--loss", default="cosine", choices=["cosine", "triplet"], help="Loss类型")
    parser_train_bi.add_argument("--pool", default="mean", choices=["cls", "mean", "max"], help="池化方式")
    parser_train_bi.add_argument("--num_hidden_layers", default=4, type=int, help="BERT层数")
    parser_train_bi.add_argument("--epochs", default=3, type=int, help="训练轮数")
    parser_train_bi.add_argument("--batch_size", default=32, type=int, help="批次大小")
    parser_train_bi.add_argument("--max_length", default=64, type=int, help="单句最大长度")
    parser_train_bi.add_argument("--margin", default=0.3, type=float, help="Loss margin")
    parser_train_bi.add_argument("--grad_accum", default=1, type=int, help="梯度累积步数")
    parser_train_bi.add_argument("--data_dir", default=DATA_DIR, type=Path)
    parser_train_bi.add_argument("--bert_path", default=BERT_PATH, type=str)

    # 3. 训练CrossEncoder
    parser_train_cross = subparsers.add_parser("train_cross", help="训练CrossEncoder")
    parser_train_cross.add_argument("--num_hidden_layers", default=4, type=int, help="BERT层数")
    parser_train_cross.add_argument("--epochs", default=3, type=int, help="训练轮数")
    parser_train_cross.add_argument("--batch_size", default=32, type=int, help="批次大小")
    parser_train_cross.add_argument("--max_length", default=128, type=int, help="拼接后最大长度")
    parser_train_cross.add_argument("--grad_accum", default=1, type=int, help="梯度累积步数")
    parser_train_cross.add_argument("--data_dir", default=DATA_DIR, type=Path)
    parser_train_cross.add_argument("--bert_path", default=BERT_PATH, type=str)

    # 4. 对比方法
    parser_compare = subparsers.add_parser("compare", help="对比不同方法")
    parser_compare.add_argument("--split", default="validation", help="评估数据集")
    parser_compare.add_argument("--batch_size", default=32, type=int, help="批次大小")
    parser_compare.add_argument("--bert_path", default=BERT_PATH, type=str)

    # 5. 分析BadCase
    parser_badcase = subparsers.add_parser("badcase", help="分析BadCase")
    parser_badcase.add_argument("--model_type", default="biencoder", choices=["biencoder", "crossencoder"], help="模型类型")
    parser_badcase.add_argument("--ckpt", default=None, type=Path, help="模型路径")
    parser_badcase.add_argument("--n_cases", default=10, type=int, help="展示示例数")
    parser_badcase.add_argument("--batch_size", default=32, type=int, help="批次大小")
    parser_badcase.add_argument("--bert_path", default=BERT_PATH, type=str)

    args = parser.parse_args()

    # 执行对应子命令
    if args.command == "explore":
        explore_data(args)
    elif args.command == "train_bi":
        train_biencoder(args)
    elif args.command == "train_cross":
        train_crossencoder(args)
    elif args.command == "compare":
        compare_methods(args)
    elif args.command == "badcase":
        analyze_badcases(args)

if __name__ == "__main__":
    main()
