"""
Week 8 文本匹配作业一键复现脚本

按顺序执行：
  1. LCQMC 上 BiEncoder(Cosine/Triplet) + CrossEncoder 训练
  2. BQ Corpus 上 BiEncoder(Cosine/Triplet) + CrossEncoder 训练
  3. 生成两个数据集的对比表格与可视化图表
  4. LCQMC / BQ Corpus 上的 Qwen2-0.5B LoRA SFT 训练与评估

默认会跳过已存在的 checkpoint；使用 --retrain 可强制重新训练。

使用方式：
  python run_all.py                # 只补充缺失的实验并生成结果
  python run_all.py --retrain      # 强制全部重新训练
  python run_all.py --datasets lcqmc  # 只跑 LCQMC
"""

import os
import argparse
import subprocess
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "outputs" / "checkpoints"
SFT_DIR_PATTERN = "{dataset}_sft_adapter_ntrain{num_train}"

BERT_METHODS = [
    ("biencoder", "cosine", 64, 64),
    ("biencoder", "triplet", 64, 64),
    ("crossencoder", None, 128, 64),
]


def run(cmd, env=None):
    """运行单条 shell 命令，失败时抛出异常。"""
    print(f"\n[RUN] {' '.join(cmd)}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, cwd=ROOT, env=full_env, check=True)


def bert_ckpt_name(dataset, model_type, loss, num_train):
    if model_type == "crossencoder":
        return f"{dataset}_crossencoder_ntrain{num_train}_best.pt"
    return f"{dataset}_biencoder_{loss}_ntrain{num_train}_best.pt"


def train_bert(dataset, num_train, retrain):
    """训练某个数据集上的三种 BERT 方法。"""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for model_type, loss, max_length, batch_size in BERT_METHODS:
        ckpt = CKPT_DIR / bert_ckpt_name(dataset, model_type, loss, num_train)
        if ckpt.exists() and not retrain:
            print(f"[SKIP] checkpoint 已存在: {ckpt.name}")
            continue

        if model_type == "biencoder":
            run([
                "python", "src/train_biencoder.py",
                "--dataset", dataset,
                "--loss", loss,
                "--num_train", str(num_train),
                "--epochs", "2",
                "--batch_size", str(batch_size),
                "--num_hidden_layers", "4",
                "--pool", "mean",
                "--max_length", str(max_length),
            ])
        else:
            run([
                "python", "src/train_crossencoder.py",
                "--dataset", dataset,
                "--num_train", str(num_train),
                "--epochs", "2",
                "--batch_size", str(batch_size),
                "--num_hidden_layers", "4",
                "--max_length", str(max_length),
            ])


def compare(dataset):
    """生成对比表格与图表。"""
    run([
        "python", "src/compare_methods.py",
        "--dataset", dataset,
        "--split", "validation",
        "--batch_size", "64",
    ])


def train_sft(dataset, num_train, retrain):
    """训练 LLM LoRA SFT。"""
    sft_dir = ROOT / "outputs" / SFT_DIR_PATTERN.format(dataset=dataset, num_train=num_train)
    if sft_dir.exists() and not retrain:
        print(f"[SKIP] SFT adapter 已存在: {sft_dir.name}")
        return

    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    run([
        "python", "src_llm/train_sft.py",
        "--dataset", dataset,
        "--num_train", str(num_train),
        "--epochs", "1",
        "--batch_size", "4",
        "--grad_accum", "4",
        "--max_length", "128",
    ], env=env)


def eval_sft(dataset, num_train):
    """评估 LLM SFT。"""
    sft_dir = ROOT / "outputs" / SFT_DIR_PATTERN.format(dataset=dataset, num_train=num_train)
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    run([
        "python", "src_llm/evaluate_sft.py",
        "--dataset", dataset,
        "--ckpt_dir", str(sft_dir),
        "--num_samples", "500",
    ], env=env)


def parse_args():
    parser = argparse.ArgumentParser(description="Week 8 文本匹配作业一键复现")
    parser.add_argument("--datasets", nargs="+", default=["lcqmc", "bq_corpus"],
                        choices=["afqmc", "lcqmc", "bq_corpus"],
                        help="要运行的数据集，默认 lcqmc + bq_corpus")
    parser.add_argument("--num_train", default=10000, type=int,
                        help="BERT 方法训练样本数")
    parser.add_argument("--num_train_sft", default=2000, type=int,
                        help="SFT 训练样本数")
    parser.add_argument("--retrain", action="store_true",
                        help="强制重新训练（覆盖已有 checkpoint）")
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"数据集: {dataset}")
        print(f"{'='*70}")

        # 1. BERT 方法
        train_bert(dataset, args.num_train, args.retrain)
        compare(dataset)

        # 2. LLM SFT
        train_sft(dataset, args.num_train_sft, args.retrain)
        eval_sft(dataset, args.num_train_sft)

    print("\n[OK] 全部实验已完成。")
    print("输出文件：")
    print("  - outputs/*_comparison_table.md")
    print("  - outputs/figures/*_acc_f1.png")
    print("  - outputs/figures/*_sim_distribution.png")
    print("  - outputs/logs/*_method_comparison.json")
    print("  - outputs/logs/*_sft_results.json")


if __name__ == "__main__":
    main()
