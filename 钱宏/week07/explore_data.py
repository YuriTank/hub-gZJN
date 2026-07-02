"""
peoples_daily test 数据集探索与可视化

针对 BIO 标注格式的数据集进行统计：
1. 解析 tokens 和 ner_tags
2. 统计各实体类型分布
3. 统计文本长度和实体长度
4. 生成可视化图表
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体，防止乱码
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs" / "peoples_figures"

# 定义 BIO 标签到中文实体类型的映射
# 注意：O 标签不计入实体统计
LABEL_MAP = {
    "B-PER": "人名",
    "I-PER": "人名",
    "B-ORG": "组织机构",
    "I-ORG": "组织机构",
    "B-LOC": "地名/地点",
    "I-LOC": "地名/地点",
}

def load_test_data() -> list:
    """加载 train.json 文件"""
    path = DATA_DIR / "train.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_entities_from_bio(tokens: list, ner_tags: list) -> list:
    """
    从 BIO 序列中提取实体片段
    返回: [(entity_text, entity_type), ...]
    """
    entities = []
    current_entity_chars = []
    current_type = None
    
    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            # 如果之前有正在收集的实体，先保存
            if current_type and current_entity_chars:
                entities.append(("".join(current_entity_chars), current_type))
            
            # 开始新实体
            current_type = LABEL_MAP.get(tag, "Unknown")
            current_entity_chars = [token]
            
        elif tag.startswith("I-"):
            # 继续当前实体
            expected_tag = tag.replace("I-", "B-")
            # 简单校验：确保 I 标签跟随对应的 B 标签类型（可选，视数据质量而定）
            if current_type and LABEL_MAP.get(expected_tag) == current_type:
                current_entity_chars.append(token)
            else:
                # 如果 I 标签不匹配当前的 B，通常视为错误或新实体的开始（取决于具体标注规范）
                # 这里保守处理：如果类型不匹配，结束上一个，尝试开始新的（如果它是合法的I开头）
                if current_entity_chars:
                    entities.append(("".join(current_entity_chars), current_type))
                current_type = LABEL_MAP.get(tag, "Unknown")
                current_entity_chars = [token]
        else:
            # O 标签或其他
            if current_entity_chars:
                entities.append(("".join(current_entity_chars), current_type))
                current_entity_chars = []
                current_type = None

    # 处理句子末尾可能残留的实体
    if current_entity_chars:
        entities.append(("".join(current_entity_chars), current_type))
        
    return entities

def collect_stats(records: list) -> dict:
    """
    收集统计数据
    """
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    all_entities_by_type = {} # 用于存储示例

    for row in records:
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        
        # 1. 文本长度 (字符数)
        text_len = len(tokens)
        text_lengths.append(text_len)
        
        # 2. 提取实体
        entities = extract_entities_from_bio(tokens, ner_tags)
        total_entities = len(entities)
        entity_per_sentence.append(total_entities)
        
        for ent_text, ent_type in entities:
            entity_type_counts[ent_type] += 1
            entity_lengths.append(len(ent_text))
            
            if ent_type not in all_entities_by_type:
                all_entities_by_type[ent_type] = []
            all_entities_by_type[ent_type].append(ent_text)

    return {
        "entity_type_counts": entity_type_counts,  # 实体类型出现的总次数 人名、组织机构、地名/地点
        "entity_lengths": entity_lengths, # 每一个实体的长度列表，如：北京长度2
        "text_lengths": text_lengths, # 样本长度列表
        "entity_per_sentence": entity_per_sentence, # 每个样本包含的实体数列表
        "all_entities_by_type": all_entities_by_type, # 实体类型对应的实体列表 key: 实体类型 value: 实体列表
    }

def print_summary(stats: dict):
    """打印统计摘要"""
    print("=" * 70)
    print("Peoples Daily Test 数据集统计摘要")
    print("=" * 70)

    n_samples = len(stats['text_lengths'])
    if n_samples == 0:
        print("数据为空")
        return

    print(f"\n【测试集概览】")
    print(f"  样本数：{n_samples} 条")
    print(f"  文本平均长度：{sum(stats['text_lengths']) / n_samples:.1f} 字")
    print(f"  文本最大长度：{max(stats['text_lengths'])} 字")
    print(f"  平均实体数/句：{sum(stats['entity_per_sentence']) / n_samples:.2f}")
    print(f"  实体总数：{sum(stats['entity_type_counts'].values())}")
    
    if stats['entity_lengths']:
        print(f"  平均实体长度：{sum(stats['entity_lengths']) / len(stats['entity_lengths']):.1f} 字")

    print(f"\n【各类实体频次分布】")
    for etype, cnt in sorted(stats["entity_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {etype:10s} : {cnt:5d} 条")

    print(f"\n【各类实体示例 (前5个)】")
    for etype in sorted(stats["all_entities_by_type"]):
        examples = list(dict.fromkeys(stats["all_entities_by_type"][etype]))[:10] # 去重取前5
        print(f"  {etype:10s} : {' | '.join(examples)}")
    print()

def plot_entity_distribution(stats: dict):
    """绘制实体类型分布图"""
    counts = stats["entity_type_counts"]
    if not counts:
        print("无实体数据，跳过实体分布绘图")
        return

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    
    # 在柱状图上方添加数值
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values)*0.01, str(v),
                ha="center", va="bottom", fontsize=10)
                
    ax.set_title("Test 集各类实体频次分布", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIG_DIR / "test_entity_distribution.png"
    fig.savefig(save_path, dpi=120)
    print(f"  已保存 → {save_path}")
    plt.close()

def plot_text_length_distribution(stats: dict):
    """绘制文本长度分布图"""
    lengths = stats["text_lengths"]
    if not lengths:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    
    # 添加平均线和中位数线
    mean_len = sum(lengths) / len(lengths)
    median_len = sorted(lengths)[len(lengths)//2]
    
    ax.axvline(x=mean_len, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_len:.1f}")
    ax.axvline(x=median_len, color="green", linestyle="--", linewidth=1.5, label=f"Median={median_len}")
    
    ax.set_title("Test 集文本长度分布", fontsize=14)
    ax.set_xlabel("文本字符数 (Tokens)")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    
    save_path = FIG_DIR / "test_text_length_distribution.png"
    fig.savefig(save_path, dpi=120)
    print(f"  已保存 → {save_path}")
    plt.close()

def plot_entity_length_distribution(stats: dict):
    """绘制实体长度分布图"""
    lengths = stats["entity_lengths"]
    if not lengths:
        return

    length_counts = Counter(lengths)
    xs = sorted(length_counts.keys())
    ys = [length_counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    # 只展示前20种长度，避免图表过宽
    display_xs = xs[:20]
    display_ys = ys[:20]
    
    ax.bar([str(x) for x in display_xs], display_ys, color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title("Test 集实体长度分布 (前20种长度)", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    
    save_path = FIG_DIR / "test_entity_length_distribution.png"
    fig.savefig(save_path, dpi=120)
    print(f"  已保存 → {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="探索 peoples_daily test 数据集")
    parser.parse_args()

    print("正在加载 test.json ...")
    try:
        test_records = load_test_data()
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print("正在统计分析 ...")
    stats = collect_stats(test_records)

    print_summary(stats)

    print("正在生成可视化图表...")
    plot_entity_distribution(stats)
    plot_text_length_distribution(stats)
    plot_entity_length_distribution(stats)

    print("\n探索完成！图表已保存到 outputs/peoples_figures/")

if __name__ == "__main__":
    main()
