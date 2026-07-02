"""
在 LCQMC 与 BQ Corpus 上试验不同文本匹配方法。

方法：
  1. char_jaccard：字符集合 Jaccard，相似度阈值分类
  2. char_tfidf_cosine：字符 n-gram TF-IDF 向量余弦相似度，验证集搜索阈值
  3. pair_tfidf_lr：句对特征 + LogisticRegression 二分类

使用方式：
  python src/experiment_other_datasets.py
  python src/experiment_other_datasets.py --datasets lcqmc bq_corpus --max_train 50000
  python src/experiment_other_datasets.py --max_train 0   # 使用全量训练集
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import paired_cosine_distances

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
LOG_DIR = ROOT / "outputs" / "logs"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def limit_rows(rows, max_rows):
    if max_rows and max_rows > 0:
        return rows[:max_rows]
    return rows


def split_xy(rows):
    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    return s1, s2, y


def evaluate_predictions(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_positive": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
    }


def find_best_threshold(scores, labels):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.0, 1.0, 101):
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
    return best_threshold


def char_jaccard_scores(s1, s2):
    scores = []
    for a, b in zip(s1, s2):
        ca, cb = set(a), set(b)
        union = ca | cb
        scores.append(len(ca & cb) / len(union) if union else 0.0)
    return np.array(scores)


def run_char_jaccard(train_s1, train_s2, train_y, val_s1, val_s2, val_y):
    train_scores = char_jaccard_scores(train_s1, train_s2)
    threshold = find_best_threshold(train_scores, train_y)
    val_scores = char_jaccard_scores(val_s1, val_s2)
    preds = (val_scores >= threshold).astype(int)
    metrics = evaluate_predictions(val_y, preds)
    metrics["threshold"] = threshold
    return metrics


def run_char_tfidf_cosine(train_s1, train_s2, train_y, val_s1, val_s2, val_y):
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, max_features=80000)
    vectorizer.fit(train_s1 + train_s2)

    train_vec1 = vectorizer.transform(train_s1)
    train_vec2 = vectorizer.transform(train_s2)
    train_scores = 1.0 - paired_cosine_distances(train_vec1, train_vec2)
    threshold = find_best_threshold(train_scores, train_y)

    val_vec1 = vectorizer.transform(val_s1)
    val_vec2 = vectorizer.transform(val_s2)
    val_scores = 1.0 - paired_cosine_distances(val_vec1, val_vec2)
    preds = (val_scores >= threshold).astype(int)
    metrics = evaluate_predictions(val_y, preds)
    metrics["threshold"] = threshold
    return metrics


def build_pair_features(vec1, vec2):
    return sparse.hstack([vec1, vec2, abs(vec1 - vec2), vec1.multiply(vec2)], format="csr")


def run_pair_tfidf_lr(train_s1, train_s2, train_y, val_s1, val_s2, val_y):
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, max_features=100000)
    vectorizer.fit(train_s1 + train_s2)

    train_vec1 = vectorizer.transform(train_s1)
    train_vec2 = vectorizer.transform(train_s2)
    val_vec1 = vectorizer.transform(val_s1)
    val_vec2 = vectorizer.transform(val_s2)

    train_features = build_pair_features(train_vec1, train_vec2)
    val_features = build_pair_features(val_vec1, val_vec2)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(train_features, train_y)
    preds = clf.predict(val_features)
    return evaluate_predictions(val_y, preds)


def run_dataset(dataset_name, max_train):
    dataset_dir = DATA_ROOT / dataset_name
    train_rows = limit_rows(load_jsonl(dataset_dir / "train.jsonl"), max_train)
    val_rows = load_jsonl(dataset_dir / "validation.jsonl")

    train_s1, train_s2, train_y = split_xy(train_rows)
    val_s1, val_s2, val_y = split_xy(val_rows)

    methods = [
        ("char_jaccard", run_char_jaccard),
        ("char_tfidf_cosine", run_char_tfidf_cosine),
        ("pair_tfidf_lr", run_pair_tfidf_lr),
    ]

    results = []
    for method_name, method_fn in methods:
        print(f"[{dataset_name}] running {method_name} ...")
        metrics = method_fn(train_s1, train_s2, train_y, val_s1, val_s2, val_y)
        result = {
            "dataset": dataset_name,
            "method": method_name,
            "train_size": len(train_rows),
            "validation_size": len(val_rows),
            **metrics,
        }
        results.append(result)
        print(
            f"  acc={result['accuracy']:.4f} "
            f"f1_weighted={result['f1_weighted']:.4f} "
            f"f1_positive={result['f1_positive']:.4f}"
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="LCQMC/BQ Corpus 轻量文本匹配方法对比")
    parser.add_argument("--datasets", nargs="+", default=["lcqmc", "bq_corpus"])
    parser.add_argument("--max_train", type=int, default=50000, help="每个数据集最多使用多少训练样本；0 表示全量")
    parser.add_argument("--output", default=str(LOG_DIR / "other_datasets_method_comparison.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = []
    for dataset_name in args.datasets:
        all_results.extend(run_dataset(dataset_name, args.max_train))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n汇总：")
    print(f"{'Dataset':<12} {'Method':<20} {'Acc':>8} {'F1(w)':>8} {'F1(pos)':>8}")
    for r in all_results:
        print(
            f"{r['dataset']:<12} {r['method']:<20} "
            f"{r['accuracy']:>8.4f} {r['f1_weighted']:>8.4f} {r['f1_positive']:>8.4f}"
        )
    print(f"\n结果已保存 → {output_path}")


if __name__ == "__main__":
    main()
