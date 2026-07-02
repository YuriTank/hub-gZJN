"""
三种文本匹配方法对比报告生成器

用法：
  python compare_report.py --dataset lcqmc

输出：
  outputs/{dataset}_comparison_table.md
  outputs/figures/{dataset}_acc_f1.png
  outputs/figures/{dataset}_sim_distribution.png
"""

import os
import re
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertTokenizer

from dataset import build_crossencoder_loaders, build_pair_loaders
from evaluate import eval_biencoder, eval_crossencoder, plot_similarity_distribution
from model import build_biencoder, build_crossencoder

ROOT = Path(__file__).parent.parent
BERT_PATH = "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR = ROOT / "outputs" / "figures"

METHODS = ["biencoder_cosine", "biencoder_triplet", "crossencoder"]


def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt


def eval_method(dataset, method, ckpt_path, device):
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    data_dir = ROOT / "data" / dataset
    ckpt = load_ckpt(ckpt_path, device)
    args = ckpt.get("args", {})

    if method.startswith("biencoder"):
        pool = args.get("pool", "mean")
        num_layers = args.get("num_hidden_layers", 4)
        model = build_biencoder(BERT_PATH, pool=pool, num_hidden_layers=num_layers).to(device)
        model.load_state_dict(ckpt["state_dict"])

        _, val_loader, _ = build_pair_loaders(
            data_dir, tokenizer,
            max_length=args.get("max_length", 64), batch_size=64,
        )
        metrics = eval_biencoder(model, val_loader, device, find_threshold=True)
        return {
            "method": method,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "threshold": metrics.get("threshold", None),
            "auc": metrics.get("auc", None),
            "similarities": metrics["similarities"],
            "labels": metrics["labels"],
        }
    else:  # crossencoder
        num_layers = args.get("num_hidden_layers", 4)
        model = build_crossencoder(BERT_PATH, num_hidden_layers=num_layers).to(device)
        model.load_state_dict(ckpt["state_dict"])

        _, val_loader, _ = build_crossencoder_loaders(
            data_dir, tokenizer,
            max_length=args.get("max_length", 128), batch_size=64,
        )
        metrics = eval_crossencoder(model, val_loader, device)
        return {
            "method": method,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "threshold": None,
            "auc": None,
            "similarities": None,
            "labels": None,
        }


def find_checkpoints(dataset):
    """根据命名约定找出每个方法的 checkpoint（优先使用 _ntrain10000_ 版本）。"""
    pattern = re.compile(
        re.escape(dataset) + r"_(biencoder_cosine|biencoder_triplet|crossencoder)(?:_ntrain(\d+))?_best\.pt$"
    )
    candidates = {}
    for p in CKPT_DIR.glob(f"{dataset}_*.pt"):
        m = pattern.match(p.name)
        if not m:
            continue
        method = m.group(1)
        ntrain = int(m.group(2)) if m.group(2) else 0
        # 保留 ntrain 最大的版本
        if method not in candidates or ntrain > candidates[method][1]:
            candidates[method] = (p, ntrain)
    return {method: pair[0] for method, pair in candidates.items()}


def plot_bar(results, dataset, save_path):
    methods = [r["method"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, accs, width, label="Accuracy")
    rects2 = ax.bar(x + width / 2, f1s, width, label="F1 (weighted)")

    ax.set_ylabel("Score")
    ax.set_title(f"{dataset.upper()} Validation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] 柱状图 -> {save_path}")


def plot_sim(results, dataset, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        if r["similarities"] is None:
            continue
        sims = np.array(r["similarities"])
        labels = np.array(r["labels"])
        pos = sims[labels == 1]
        neg = sims[labels == 0]
        ax.hist(neg, bins=50, alpha=0.5, label=f"{r['method']} neg", density=True)
        ax.hist(pos, bins=50, alpha=0.5, label=f"{r['method']} pos", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset.upper()} Similarity Distribution")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] 相似度分布图 -> {save_path}")


def build_markdown(results, dataset, figure_paths):
    lines = []
    lines.append(f"## {dataset.upper()} 三种方法对比")
    lines.append("")
    lines.append("| 方法 | Accuracy | F1 (weighted) | Threshold | AUC-ROC |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        thr = f"{r['threshold']:.2f}" if r["threshold"] is not None else "-"
        auc = f"{r['auc']:.4f}" if r["auc"] is not None else "-"
        lines.append(
            f"| {r['method']} | {r['accuracy']:.4f} | {r['f1']:.4f} | {thr} | {auc} |"
        )
    lines.append("")
    for name, path in figure_paths.items():
        rel = path.relative_to(ROOT / "outputs")
        lines.append(f"- {name}: `{rel}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["afqmc", "lcqmc", "bq_corpus"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare_report] dataset={args.dataset}  device={device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpts = find_checkpoints(args.dataset)
    print(f"发现 checkpoint: {ckpts}")

    results = []
    for method in METHODS:
        if method not in ckpts:
            print(f"[X] 缺少 {method} 的 checkpoint，跳过")
            continue
        print(f"\n评估 {method} ...")
        res = eval_method(args.dataset, method, ckpts[method], device)
        results.append(res)
        print(f"  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}")

    if not results:
        print("[X] 没有可评估的结果")
        return

    bar_path = FIG_DIR / f"{args.dataset}_acc_f1.png"
    sim_path = FIG_DIR / f"{args.dataset}_sim_distribution.png"
    plot_bar(results, args.dataset, bar_path)
    plot_sim(results, args.dataset, sim_path)

    md = build_markdown(
        results, args.dataset,
        {"柱状图": bar_path, "相似度分布图": sim_path}
    )
    md_path = ROOT / "outputs" / f"{args.dataset}_comparison_table.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[OK] 对比表格 -> {md_path}")


if __name__ == "__main__":
    main()
