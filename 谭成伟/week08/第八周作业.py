"""
文本匹配方法全面对比脚本
读取 BERT 和 LLM 的结果文件，生成对比表格、柱状图和分析结论
"""
import os
import json

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径配置 ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT_RESULT_PATH = os.path.join(BASE_DIR, "homework", "outputs", "bert_results.json")
LLM_RESULT_PATH = os.path.join(BASE_DIR, "homework", "outputs", "llm_results.json")
FIGURE_DIR = os.path.join(BASE_DIR, "homework", "outputs", "figures")
FIGURE_PATH = os.path.join(FIGURE_DIR, "all_methods_comparison.png")

DATASETS = ["bq_corpus", "lcqmc"]

# 方法定义：(结果文件, 内部key, 显示名)
METHODS = [
    ("bert", "biencoder_cosine",  "BiEncoder+CosineEmbeddingLoss"),
    ("bert", "biencoder_triplet", "BiEncoder+TripletLoss"),
    ("bert", "crossencoder",      "CrossEncoder+CrossEntropyLoss"),
    ("llm",  "lora_sft",         "LLM LoRA SFT"),
    ("llm",  "full_ft_sft",      "LLM Full FT SFT"),
]


# ── 数据读取 ──────────────────────────────────────────────────────────────
def load_results():
    bert_data = None
    llm_data = None

    if os.path.exists(BERT_RESULT_PATH):
        with open(BERT_RESULT_PATH, "r", encoding="utf-8-sig") as f:
            bert_data = json.load(f)
        print(f"[INFO] 已加载 BERT 结果: {BERT_RESULT_PATH}")
    else:
        print(f"[WARNING] BERT 结果文件不存在: {BERT_RESULT_PATH}")

    if os.path.exists(LLM_RESULT_PATH):
        with open(LLM_RESULT_PATH, "r", encoding="utf-8-sig") as f:
            llm_data = json.load(f)
        print(f"[INFO] 已加载 LLM 结果: {LLM_RESULT_PATH}")
    else:
        print(f"[WARNING] LLM 结果文件不存在: {LLM_RESULT_PATH}")

    return bert_data, llm_data


def get_method_metrics(bert_data, llm_data, dataset, method_info):
    """获取某个方法在某个数据集上的指标，返回 (accuracy, f1, 备注)"""
    source, key, _ = method_info
    data = bert_data if source == "bert" else llm_data

    if data is None or dataset not in data or key not in data[dataset]:
        return None, None, "无数据"

    entry = data[dataset][key]
    accuracy = entry.get("accuracy")
    f1 = entry.get("f1") or entry.get("f1_pos")

    # 备注
    if source == "bert":
        if key == "crossencoder":
            note = "argmax"
        else:
            threshold = entry.get("threshold")
            note = f"threshold={threshold:.2f}" if threshold is not None else "-"
    else:
        parse_fail = entry.get("parse_fail", 0)
        note = f"parse_fail={parse_fail}"

    return accuracy, f1, note


# ── 打印对比表格 ──────────────────────────────────────────────────────────
def print_comparison_table(bert_data, llm_data):
    header = "                    文本匹配方法全面对比（bq_corpus / lcqmc）"
    top_bottom = "═" * 75
    separator = "─" * 75

    print(top_bottom)
    print(header)
    print(top_bottom)
    print(f"{'数据集':<12}│{'方法':<36}│{'Accuracy':<10}│{'F1':<10}│{'备注'}")
    print(separator)

    for dataset in DATASETS:
        for method_info in METHODS:
            source, key, display_name = method_info
            accuracy, f1, note = get_method_metrics(bert_data, llm_data, dataset, method_info)

            acc_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"

            print(f"{dataset:<12}│{display_name:<36}│{acc_str:<10}│{f1_str:<10}│{note}")
        print(separator)

    print(top_bottom)


# ── 绘制柱状图 ───────────────────────────────────────────────────────────
def plot_comparison_chart(bert_data, llm_data):
    os.makedirs(FIGURE_DIR, exist_ok=True)

    method_names = [m[2] for m in METHODS]
    n_methods = len(METHODS)

    # 收集数据
    acc_data = {}
    f1_data = {}
    for dataset in DATASETS:
        accs, f1s = [], []
        for method_info in METHODS:
            accuracy, f1, _ = get_method_metrics(bert_data, llm_data, dataset, method_info)
            accs.append(accuracy if accuracy is not None else 0)
            f1s.append(f1 if f1 is not None else 0)
        acc_data[dataset] = accs
        f1_data[dataset] = f1s

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    bar_width = 0.3
    x = np.arange(n_methods)

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        accs = acc_data[dataset]
        f1s = f1_data[dataset]

        bars_acc = ax.bar(x - bar_width / 2, accs, bar_width, label="Accuracy", color="#4C72B0", edgecolor="white")
        bars_f1 = ax.bar(x + bar_width / 2, f1s, bar_width, label="F1", color="#DD8452", edgecolor="white")

        # 在柱子上方标注数值
        for bar in bars_acc:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8)
        for bar in bars_f1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8)

        dataset_label = "bq_corpus（金融领域）" if dataset == "bq_corpus" else "lcqmc（开放领域）"
        ax.set_title(dataset_label, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("分数", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("文本匹配方法全面对比", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"[INFO] 对比柱状图已保存至: {FIGURE_PATH}")
    plt.close()


# ── 打印分析结论 ──────────────────────────────────────────────────────────
def print_analysis(bert_data, llm_data):
    print("\n" + "=" * 75)
    print("                        分析结论")
    print("=" * 75)

    # 收集数据用于分析
    all_metrics = {}
    for dataset in DATASETS:
        all_metrics[dataset] = {}
        for method_info in METHODS:
            source, key, display_name = method_info
            accuracy, f1, note = get_method_metrics(bert_data, llm_data, dataset, method_info)
            all_metrics[dataset][display_name] = {"accuracy": accuracy, "f1": f1, "note": note}

    # 1. BERT 3种训练方式对比
    print("\n【1. BERT 3种训练方式对比：哪种Loss最好，为什么】")
    bert_methods = ["BiEncoder+CosineEmbeddingLoss", "BiEncoder+TripletLoss", "CrossEncoder+CrossEntropyLoss"]
    for dataset in DATASETS:
        print(f"\n  数据集: {dataset}")
        best_acc_method = None
        best_acc_val = -1
        best_f1_method = None
        best_f1_val = -1
        for m in bert_methods:
            info = all_metrics[dataset].get(m, {})
            acc = info.get("accuracy") or 0
            f1 = info.get("f1") or 0
            print(f"    {m}: Accuracy={acc:.4f}, F1={f1:.4f}")
            if acc > best_acc_val:
                best_acc_val = acc
                best_acc_method = m
            if f1 > best_f1_val:
                best_f1_val = f1
                best_f1_method = m
        print(f"    → Accuracy最优: {best_acc_method} ({best_acc_val:.4f})")
        print(f"    → F1最优: {best_f1_method} ({best_f1_val:.4f})")

    print("""
  分析:
  - CrossEncoder 通常表现最好，因为它同时编码两个句子的交互信息，
    能捕捉更细粒度的语义关系，直接输出匹配概率。
  - BiEncoder+TripletLoss 通常优于 BiEncoder+CosineEmbeddingLoss，
    因为 TripletLoss 使用三元组（anchor, positive, negative）进行训练，
    明确拉近正例、推远负例，比 CosineEmbeddingLoss 的成对比较更有效。
  - BiEncoder 的优势在于可以预计算句向量，推理速度更快，
    适合大规模检索场景；CrossEncoder 则适合精排阶段。""")

    # 2. LLM 2种训练方式对比
    print("\n【2. LLM 2种训练方式对比：LoRA vs Full FT】")
    llm_methods = ["LLM LoRA SFT", "LLM Full FT SFT"]
    for dataset in DATASETS:
        print(f"\n  数据集: {dataset}")
        for m in llm_methods:
            info = all_metrics[dataset].get(m, {})
            acc = info.get("accuracy") or 0
            f1 = info.get("f1") or 0
            note = info.get("note", "")
            print(f"    {m}: Accuracy={acc:.4f}, F1={f1:.4f} ({note})")

    print("""
  分析:
  - Full FT（全量微调）通常比 LoRA 效果更好，因为它更新所有参数，
    模型能更充分地适应下游任务。
  - 但 Full FT 显存开销大、训练时间长，且在小数据集上更容易过拟合。
  - LoRA 只更新低秩增量矩阵，参数量少（通常 <1%），训练效率高，
    且在数据量有限时泛化能力可能更好。
  - 如果 parse_fail 较多，说明模型的输出格式不够稳定，
    Full FT 可能更稳定，LoRA 可能需要更多格式约束。""")

    # 3. BERT vs LLM 对比
    print("\n【3. BERT vs LLM 对比：判别式 vs 生成式】")
    for dataset in DATASETS:
        print(f"\n  数据集: {dataset}")
        bert_best_acc = 0
        llm_best_acc = 0
        for m in bert_methods:
            info = all_metrics[dataset].get(m, {})
            acc = info.get("accuracy") or 0
            bert_best_acc = max(bert_best_acc, acc)
        for m in llm_methods:
            info = all_metrics[dataset].get(m, {})
            acc = info.get("accuracy") or 0
            llm_best_acc = max(llm_best_acc, acc)
        print(f"    BERT 最优 Accuracy: {bert_best_acc:.4f}")
        print(f"    LLM  最优 Accuracy: {llm_best_acc:.4f}")

    print("""
  分析:
  - BERT（判别式模型）在文本匹配任务上通常效果更好且更稳定，
    因为它直接建模匹配概率，目标函数与任务完全一致。
  - LLM（生成式模型）需要将匹配判断转化为文本生成任务，
    存在输出解析失败的风险（parse_fail），且推理速度较慢。
  - LLM 的优势在于：零样本/少样本能力强，无需专门训练即可处理；
    且可以通过 Prompt 灵活调整任务定义，泛化到多种 NLU 任务。
  - 在工业落地中，BERT 类模型仍是文本匹配的首选，
    LLM 更适合作为通用基座或处理复杂推理场景。""")

    # 4. 两个数据集的差异
    print("\n【4. 两个数据集的差异：bq_corpus（金融领域）vs lcqmc（开放领域）】")
    for m in [m[2] for m in METHODS]:
        bq_info = all_metrics["bq_corpus"].get(m, {})
        lcqmc_info = all_metrics["lcqmc"].get(m, {})
        bq_acc = bq_info.get("accuracy") or 0
        lcqmc_acc = lcqmc_info.get("accuracy") or 0
        diff = bq_acc - lcqmc_acc
        sign = "+" if diff > 0 else ""
        print(f"  {m}: bq={bq_acc:.4f}, lcqmc={lcqmc_acc:.4f}, 差值={sign}{diff:.4f}")

    print("""
  分析:
  - bq_corpus 是金融领域的问答匹配数据集，专业术语多、语义差异更细微，
    模型需要理解领域知识才能准确判断。
  - lcqmc 是开放领域的日常问答匹配数据集，语义差异相对明显，
    模型更容易区分匹配与不匹配的句子对。
  - 通常 lcqmc 上的准确率更高，因为开放领域的匹配模式更通用；
    bq_corpus 上模型需要更强的领域适应能力。
  - 对于领域数据，领域预训练或领域微调可以显著提升效果；
    LLM 由于预训练语料广泛，可能在领域数据上有一定优势。""")

    print("=" * 75)


# ── 主函数 ────────────────────────────────────────────────────────────────
def main():
    bert_data, llm_data = load_results()

    if bert_data is None and llm_data is None:
        print("[ERROR] 没有找到任何结果文件，无法进行对比。")
        return

    print()
    print_comparison_table(bert_data, llm_data)
    plot_comparison_chart(bert_data, llm_data)
    print_analysis(bert_data, llm_data)


if __name__ == "__main__":
    main()
